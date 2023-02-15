from src.config import get_params
import matplotlib.pyplot as plt
import os, random, subprocess
from stable_baselines3 import A2C, PPO, SAC, TD3
from sb3_contrib import ARS
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.results_plotter import load_results, ts2xy
#from src.REINFORCE.REINFORCE import REINFORCE
from src.dataloader import get_sentiment_src_tgt_dataloader, get_sentiment_unseen_dataloaders, all_amazon_domains, load_amazon_processed, Dataset,collate_fn_for_sentiment
from src.model import CNN
from src.env import MESS_Sequential_Env
from src.trainer import Trainer
from src.conll2002_measure import *
import torch
import numpy as np

from torchtext.data.functional import numericalize_tokens_from_iterator
from torch.utils.data import DataLoader


def random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    set_random_seed(seed)
    torch.backends.cudnn.deterministic = True



def train(params):
    
    if not os.path.isdir('log'):
        subprocess.Popen(['mkdir','log'])
    print('Load amazon dataset...')
    
    raw_train, pretrainloader, trainloader , tgt_devloader, tgt_testloader, vocab =  get_sentiment_src_tgt_dataloader(params.tgt_dm, params.batch_size, params.bag_size)
    unseen_domains_dataloader = get_sentiment_unseen_dataloaders('data/sentiment/sorted_data', vocab, params.batch_size)
    print('Init sentiment classifier...')
    model = CNN(params, vocab)
    eval_model = CNN(params, vocab)
    
    
    model.cuda()
    eval_model.cuda()
    trainer = Trainer(model, params )
    print("Pretraining on source dataset ...")
    trainer.train(pretrainloader,  tgt_devloader, epoch=params.pretrain_epoch, is_pretrain=True)
    #trainer.train(pretrainloader,  unseen_domains_dataloader['gourmet_food'], epoch=params.pretrain_epoch, is_pretrain=True)


    print('Initialize environment...')
    
    env = MESS_Sequential_Env(
        params,
        trainer = trainer,
        train_dataloader = trainloader,
        text_x_train = raw_train,
        dev_dataloader = tgt_devloader,
    )
    #logger.info('Checking environment...')
    #check_env(env)
    env = Monitor(env, params.log_dir)

    
    print('Jointly training RL agent and nlp model...')
    
    if params.alg == 'reinforce':
        rl = REINFORCE("MlpPolicy", env, verbose=1, learning_rate=7e-4, seed=params.seed)
    elif params.alg == 'a2c':
        rl = A2C("MlpPolicy", env, verbose=1, learning_rate=7e-4, seed=params.seed)
    elif params.alg == 'sac':
        rl = SAC("MlpPolicy", env, verbose=1, seed=params.seed)
        
    

    
    rl.learn(total_timesteps=params.rl_timesteps)
    print('Save rl model...')
    rl.save(os.path.join(params.dump_path, '{}_model'.format(params.alg)))

 
    def select_data():
        feature_extractor = CNN(params, vocab)
        feature_extractor.cuda()
        feature_extractor.load_state_dict(torch.load(os.path.join(params.dump_path, 'best_finetune_model.pth')))
        
        #rl = A2C.load(os.path.join(params.dump_path, '{}_model'.format(params.alg)))
        all_sel_indices = []
        for indices, feats, labels in trainloader:
            feats = feats.cuda()
            obs = feature_extractor.get_feature(feats)
            action, _ = rl.predict(obs.detach().cpu().numpy())
            sel_indices = [i for i in range(len(action)) if action[i] > 0.5]
            all_sel_indices.extend(indices[sel_indices])
        with open('{}_{}_sel_indices.txt'.format(params.task,params.alg),'w') as f:
            for i in all_sel_indices:
                f.write(str(i)+'\n')
        print('select {} samples from source domain'.format(len(all_sel_indices)))
        return np.array(all_sel_indices)
        
        
    select_ids = select_data()
    source, target = load_amazon_processed(base_dir='./data/sentiment/amazon-reviews/processed_acl',target_domain =params.tgt_dm)
    train_inputs, train_labels = [], source['labels']
    trainiter = numericalize_tokens_from_iterator(vocab, source['reviews'])
    for seqs in trainiter:
        train_inputs.append([num for num in seqs])
    train_inputs = np.array(train_inputs)
    train_labels = np.array(train_labels)
    select_train_inputs = train_inputs[select_ids]
    select_train_labels = train_labels[select_ids]
    seltrain_dataloader = DataLoader(dataset=Dataset(np.arange(len(select_train_inputs)),select_train_inputs, select_train_labels), batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn_for_sentiment)
    #plot_results(params.log_dir)
    eval_trainer = Trainer(eval_model, params )
    eval_trainer.train(seltrain_dataloader,  tgt_devloader, epoch=20, is_pretrain=True)

    eval_model.load_state_dict(torch.load('pretrained_model.pth'))
    in_loss, in_acc = eval_trainer.evaluate_sentiment(pretrainloader)
    test_loss, test_acc = eval_trainer.evaluate_sentiment(tgt_testloader)
    print('Training finished. ')
    print(f'\t Test. Loss: {test_loss:.3f} |  Test. Acc: {test_acc*100:.2f}% ')
    print(f'\t in. Loss: {in_loss:.3f} |  in. Acc: {in_acc*100:.2f}% ')
    for domain in all_amazon_domains:
      dl = unseen_domains_dataloader[domain]
      test_loss, test_acc = eval_trainer.evaluate_sentiment(dl)
      #print('Domain: {}, Test Loss: {}, Test Acc {}'.format(domain, test_loss, test_acc))
      print('{}, test_acc: {}'.format(domain, test_acc))

if __name__ == "__main__":
    params = get_params()
 
    random_seed(params.seed)
    train(params)