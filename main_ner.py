from src.config import get_params
#from sentence_transformers import SentenceTransformer
from stable_baselines3 import A2C, SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.env_checker import check_env
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from torch.utils.data import DataLoader
from src.REINFORCE.REINFORCE import REINFORCE
from src.dataloader import Dataset, read_ner, get_conll2003_dataloader, get_ner_target_dataloader, collate_fn
from src.model import BertTagger,CNN
from src.env import MESS_Sequential_Env
from src.trainer import Trainer
from src.conll2002_measure import *
import torch
import numpy as np
import random, subprocess
import os


def make_env(env, rank, seed=0):
    """
    Utility function for multiprocessed env.

    :param env_id: (str) the environment ID
    :param num_env: (int) the number of environments you wish to have in subprocesses
    :param seed: (int) the inital seed for RNG
    :param rank: (int) index of the subprocess
    """
    def _init():
        env.seed(seed + rank)
        return env
    set_random_seed(seed)
    return _init

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
    #build two elements of environment : 1. data 2.nlp model
    print('Load NER data...')
    raw_conll, conll_pretrain, conll_train, conll_dev, conll_test = get_conll2003_dataloader(params.batch_size, params.bag_size)
    pol_devloader, pol_testloader =  get_ner_target_dataloader('politics', params.batch_size)
    sci_devloader, sci_testloader =  get_ner_target_dataloader('science', params.batch_size)
    mus_devloader, mus_testloader =  get_ner_target_dataloader('music', params.batch_size)
    lit_devloader, lit_testloader =  get_ner_target_dataloader('literature', params.batch_size)
    ai_devloader, ai_testloader =  get_ner_target_dataloader('ai', params.batch_size)
    
    print('Init bert tagger...')  
    model = BertTagger(params)
    model.cuda()
    trainer = Trainer(model, params )
    
    print("Pretraining on source dataset ...")
    trainer.train(conll_pretrain,  pol_devloader, epoch=params.pretrain_epoch, is_pretrain=True)
    print('Initialize environment...')
    
    env = MESS_Sequential_Env(
        params,
        trainer = trainer,
        train_dataloader = conll_train,
        text_x_train = raw_conll,
        dev_dataloader = pol_devloader,
    )
    env = Monitor(env, params.log_dir)


    
    #train RL model
    print('Training RL agent ...')
    if params.alg == 'reinforce':
        rl = REINFORCE("MlpPolicy", env, verbose=1, seed=params.seed)
    elif params.alg == 'a2c':
        rl = A2C("MlpPolicy", env, verbose=1, seed=params.seed)
    elif params.alg == 'sac':
        rl = SAC("MlpPolicy", env, verbose=1, seed=params.seed, learning_rate=7e-4)
    
    rl.learn(total_timesteps=params.rl_timesteps)
    print('Training finished. Save rl model...')
    rl.save(os.path.join(params.dump_path, '{}_{}_model'.format(params.task,params.alg)))
    #stats_path = os.path.join(params.log_dir, "vec_normalize.pkl")
    #env.save(stats_path)
    #stats_path = os.path.join(params.log_dir, "vec_normalize.pkl")
    def select_data():
        feature_extractor = BertTagger(params)
        feature_extractor.cuda()
        feature_extractor.load_state_dict(torch.load(os.path.join(params.dump_path, 'best_finetune_model.pth')))
        
        
        #rl = A2C.load(os.path.join(params.dump_path, '{}_{}_model'.format(params.task,params.alg)))
        all_sel_indices = []
        obs = env.reset()
        for k in range(140):
            action, _states = rl.predict(obs)
            sel_indices = [i for i in range(len(action)) if action[i] > 0.5]
            #sel_indices = [i for i in range(len(action[0])) if action[0][i] == 1]
            indices = np.arange(k*params.bag_size, (k+1)*params.bag_size)
            all_sel_indices.extend(indices[sel_indices])
            obs, rewards, dones, info = env.step(action)
        with open('{}_sel_indices.txt'.format(params.alg),'w') as f:
            for i in all_sel_indices:
                f.write(str(i)+'\n')
        print('select {} samples from source domain'.format(len(all_sel_indices)))
        return np.array(all_sel_indices)


    
    #select data using trained RL model
    select_ids = select_data()
    #with open('ppo_conll_sel_indices.txt','r') as f:
    #    lines = f.readlines()
    #select_ids = np.array([int(i) for i in lines])
    print(len(select_ids))
    #build dataloader from selected data
    _, inputs_train, labels_train = read_ner("data/ner_data/conll2003/train.txt")
    sel_train, sel_labels = np.array(inputs_train)[select_ids], np.array(labels_train)[select_ids]
    dataset_sel = Dataset(np.arange(len(sel_train)), sel_train, sel_labels)
    dataloader_sel = DataLoader(dataset=dataset_sel, batch_size=params.batch_size, shuffle=True, collate_fn=collate_fn)
    
    #train a nlp model from scratch
    print('Training NLP model ... ')
    eval_model = BertTagger(params)
    eval_model.cuda()
    eval_trainer = Trainer(eval_model, params )
    eval_trainer.train(dataloader_sel, pol_devloader, epoch=20, is_pretrain=True)
    eval_trainer.model.load_state_dict(torch.load('pretrained_model.pth'))
    conll_test_f1 = eval_trainer.evaluate_ner(conll_test)
    print(f'\t Conll Test. F1: {conll_test_f1:.3f} ')
    pol_test_f1 = eval_trainer.evaluate_ner(pol_testloader)
    print(f'\t Politics Test. F1: {pol_test_f1:.3f} ')
    sci_test_f1 = eval_trainer.evaluate_ner(sci_testloader)
    print(f'\t Science Test. F1: {sci_test_f1:.3f} ')
    mus_test_f1 = eval_trainer.evaluate_ner(mus_testloader)
    print(f'\t Music Test. F1: {mus_test_f1:.3f} ')
    lit_test_f1 = eval_trainer.evaluate_ner(lit_testloader)
    print(f'\t Literature Test. F1: {lit_test_f1:.3f} ')
    ai_test_f1 = eval_trainer.evaluate_ner(ai_testloader)
    print(f'\t AI Test. F1: {ai_test_f1:.3f} ')



if __name__ == "__main__":
    params = get_params()

    random_seed(params.seed)
    train(params)