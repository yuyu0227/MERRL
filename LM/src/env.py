import gym
import os, subprocess
from gym import spaces
import numpy as np
from fairseq.models.transformer_lm import TransformerLanguageModel
from src.reward import *
import torch
from fairseq.models import BaseFairseqModel
from fairseq.models.bart import BARTModel
class MESS_Env_Fairseq_LM(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, params, nlp_model, text_x_train, dev_data = None):
    super(NEAC_Env_Fairseq_LM, self).__init__()
    self.params = params
    self.dump_path = params.dump_path
    self.src = params.src_domain


    self.nlp_model = nlp_model
    self.dev_data = dev_data

    self.bag_size = params.bag_size
    self.bag_ids = None
    self.num_instances = len(text_x_train)
    self.text_x_train = np.array(text_x_train)
    self.shuffled_indices = np.arange(self.num_instances)

    self.reward_mode = params.reward_mode
    self.cur_step = 0
    self.cur_epoch = 0
    self.best_epoch = 0
    self.best_timestep = 0
    self.best_ppl = 500000
    self.num_steps = self.num_instances//self.bag_size
    
    self.shannon_entropy, self.renyi_entropy, self.min_entropy = np.array(compute_unigram_entropy(text_x_train))


    self.action_space=spaces.MultiBinary(self.bag_size)
    self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=
                    (self.bag_size, self.params.fairseq_hidden_dim), dtype=np.float32)
    
    

  def step(self, action):
    # Execute one time step within the environment
    reward = self._get_reward(action)
    val_score = self._take_action(action)
    nxt = self._next_state()
    done = self.cur_step == self.num_steps
    if done:
      self.cur_epoch += 1
    return nxt, reward, done, {'val_score':val_score}

  def _next_state(self):
    self.cur_step += 1
    nxt_bag_ids = self.prepare_directory(self.cur_epoch, self.cur_step, self.shuffled_indices)
    self.bag_ids = nxt_bag_ids
    self.cur_bag_emb = self.get_feature(self.nlp_model, self.bag_ids)
    return self.cur_bag_emb

  def _get_reward(self, action):
    sel_indices = [i for i in range(len(action)) if action[i] == 1]

    if self.reward_mode == 'shannon_entropy':
      sum_entropy = sum(self.shannon_entropy[self.bag_ids[sel_indices]])
      return sum_entropy
    elif self.reward_mode == 'renyi_entropy':
      sum_entropy = sum(self.renyi_entropy[self.bag_ids[sel_indices]])
      return sum_entropy
    elif self.reward_mode == 'min_entropy':
      sum_entropy = sum(self.min_entropy[self.bag_ids[sel_indices]])
      return sum_entropy
    
    
      

  def _take_action(self, action):
    sel_indices = [i for i in range(len(action)) if action[i] == 1]

    tmp_dir = 'E{}S{}'.format(self.cur_epoch,self.cur_step)
    #raw_dir = './raw/'+tmp_dir
    raw_dir = '/root/yu/raw/'+tmp_dir
    if not os.path.isdir(raw_dir):
      subprocess.Popen(['mkdir',raw_dir])
    bag_sents = self.text_x_train[self.bag_ids]
    f = open(raw_dir+'/train.tokens','w')
    for i in sel_indices:
        f.write(bag_sents[i])
    f.close()
    '''with open(raw_dir+'/train.tokens','w') as fs:
      for i in sel_indices:
        fs.write(bag_sents[i])'''
    
    train_log = open('log/train/{}.txt'.format(tmp_dir), 'w')
    subprocess.call('bash src/train_lm_one_epoch.sh {}/train.tokens data-bin-{}/{} data-bin-{}/{} ./checkpoint_best.pt'.format(raw_dir,self.src, tmp_dir,self.src,tmp_dir),shell=True,stdout=train_log)

    with open('./log/train/E{}S{}.txt'.format(self.cur_epoch,self.cur_step),'r') as f:
      lines = f.readlines()
      if 'valid' in lines[-6].split():
        val_ppl = float(lines[-6].split()[-1])
      elif 'valid' in lines[-7].split():
        val_ppl = float(lines[-7].split()[-1])
    print('\tE{}S{}:  val_ppl {}'.format(self.cur_epoch,self.cur_step,val_ppl))
    if val_ppl < self.best_ppl:
      self.best_ppl = val_ppl
      self.best_epoch = self.cur_epoch
      self.best_timestep = self.cur_step
    
    self.nlp_model = TransformerLanguageModel.from_pretrained('data-bin-{}/{}'.format(self.src,tmp_dir), 'checkpoint_best.pt', tokenizer='moses')
    return val_ppl

  def reset(self):
    # Reset the state of the environment to an initial state
    self.cur_step = 0
    self.shuffled_indices = np.arange(self.num_instances)
    np.random.shuffle(self.shuffled_indices)
    self.shannon_entropy = self.shannon_entropy[self.shuffled_indices]
    self.renyi_entropy = self.renyi_entropy[self.shuffled_indices]
    self.min_entropy = self.min_entropy[self.shuffled_indices]
    self.bag_ids = self.prepare_directory(self.cur_epoch,0,self.shuffled_indices)

    


    self.cur_bag_emb = self.get_feature(self.nlp_model, self.bag_ids)
    return self.cur_bag_emb


  def render(self, mode='human', close=False):
    # Render the environment to the screen
    pass

  def get_feature_dev(self, fairseq_model, dev_data):
    sents_emb = []
    state = []
    for txt in dev_data:
      output = fairseq_model.models[0](fairseq_model.encode(txt).unsqueeze(0))[1]['inner_states'][-1].squeeze(1).detach().numpy()
      state.append(output)
      #get sentence embedding by mean pooling each word embedding
      sents_emb.append(np.mean(output, axis=0)) 
    sents_emb = np.array(sents_emb)
    return sents_emb
  def get_feature(self, fairseq_model, sentence_ids):
    sents_emb = []
    state = []
    for idx in sentence_ids:
      output = fairseq_model.models[0](fairseq_model.encode(self.text_x_train[idx]).unsqueeze(0))[1]['inner_states'][-1].squeeze(1).detach().numpy()
      state.append(output)
      #get sentence embedding by mean pooling each word embedding
      sents_emb.append(np.mean(output, axis=0)) 
    sents_emb = np.array(sents_emb)
    return sents_emb

  def prepare_directory(self, epoch_count,timestep,shuffled_indices):
    #create directory for current timestep
    save_dir = 'data-bin-{}/E{}S{}'.format(self.src, epoch_count,timestep)
    if not os.path.isdir(save_dir):
      subprocess.Popen(['mkdir',save_dir])
  
    #remove previous step's directory
    '''if os.path.isdir('E{}S{}'.format(episode_count,timestep-1)):
      subprocess.Popen(['rm','-r','E{}S{}'.format(episode_count,timestep-1)])
    if os.path.isdir('E{}S{}'.format(episode_count-1,max_steps_per_episode-1)):
    subprocess.Popen(['rm','-r','E{}S{}'.format(episode_count-1,max_steps_per_episode-1)])'''
  

    #save ids of cur bag of sentences
    start_idx = timestep * self.bag_size
    end_idx = start_idx + self.bag_size
    sentence_ids = np.array(shuffled_indices[start_idx:end_idx])
    #np.save('sent_ids.npy',sentence_ids)

    pretrain_data_bin = 'data-bin-{}/{}'.format(self.src,self.src)
    #copy valid data to dir
    subprocess.Popen(['cp',os.path.join(pretrain_data_bin,'dict.txt'),save_dir])
    subprocess.Popen(['cp',os.path.join(pretrain_data_bin,'valid.bin'),save_dir])
    subprocess.Popen(['cp',os.path.join(pretrain_data_bin,'valid.idx'),save_dir])
    subprocess.Popen(['cp',os.path.join(pretrain_data_bin,'test.bin'),save_dir])
    subprocess.Popen(['cp',os.path.join(pretrain_data_bin,'test.idx'),save_dir])
    return sentence_ids

class NEAC_Env_Fairseq_NMT(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, params, nlp_model, text_x_src, text_x_tgt):
    super(NEAC_Env_Fairseq_NMT, self).__init__()
    self.params = params
    self.dump_path = params.dump_path
    self.nlp_model = nlp_model

    self.bag_size = params.bag_size
    self.bag_ids = None
    self.num_instances = len(text_x_tgt)
    self.text_x_src = np.array(text_x_src)
    self.text_x_tgt = np.array(text_x_tgt)
    self.shuffled_indices = np.arange(self.num_instances)

    self.reward_mode = params.reward_mode
    self.cur_step = 0
    self.cur_epoch = 0
    self.best_epoch = 0
    self.best_timestep = 0
    self.best_bleu = 0
    self.num_steps = self.num_instances//self.bag_size
    
    self.reviews_entropy = np.array(compute_unigram_entropy(text_x_src))


    self.action_space=spaces.MultiBinary(self.bag_size)
    self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=
                    (self.bag_size, self.params.fairseq_hidden_dim), dtype=np.float32)
    
    

  def step(self, action):
    # Execute one time step within the environment
    reward = self._get_reward(action)
    val_score = self._take_action(action)
    nxt = self._next_state()
    done = self.cur_step == self.num_steps
    if done:
      self.cur_epoch += 1
    return nxt, reward, done, {'val_score':val_score}

  def _next_state(self):
    self.cur_step += 1
    nxt_bag_ids = self.prepare_directory(self.cur_epoch, self.cur_step, self.shuffled_indices)
    self.bag_ids = nxt_bag_ids
    self.cur_bag_emb = self.get_feature(self.nlp_model, self.bag_ids)
    return self.cur_bag_emb

  def _get_reward(self, action):
    sel_indices = [i for i in range(len(action)) if action[i] == 1]
    if self.reward_mode == 'normalized':
      sum_entropy = sum(self.reviews_entropy[self.bag_ids[sel_indices]])
      return sum_entropy/len(sel_indices)
    elif self.reward_mode == 'ngram_entropy':
      sum_entropy = sum(self.reviews_entropy[self.bag_ids[sel_indices]])
      return sum_entropy
    elif self.reward_mode == 'dispersion':
      return get_dispersion(self.cur_bag_emb[sel_indices])
    elif self.reward_mode == 'convex_hull_volume':
      return get_convex_hull_volume(self.cur_bag_emb[sel_indices])
    elif self.reward_mode == 'graph_entropy':
      return get_graph_entropy(self.cur_bag_emb[sel_indices])
    elif self.reward_mode == 'ensemble':
      ge = get_graph_entropy(self.cur_bag_emb[sel_indices])
      ensemble = ge + 0.1 * sum_entropy
      return ensemble
    
    
      

  def _take_action(self, action):
    sel_indices = [i for i in range(len(action)) if action[i] == 1]

    tmp_dir = 'E{}S{}'.format(self.cur_epoch,self.cur_step)
    raw_dir = 'raw/'+tmp_dir
    if not os.path.isdir(raw_dir):
      subprocess.Popen(['mkdir',raw_dir])
    bag_srcs = self.text_x_src[self.bag_ids]
    bag_tgts = self.text_x_tgt[self.bag_ids]
    fs = open(raw_dir+'/train.fr','w')
    ft = open(raw_dir+'/train.en','w')
    for i in sel_indices:
      fs.write(bag_srcs[i])
      ft.write(bag_tgts[i])
    fs.close()
    ft.close()
    '''with open(raw_dir+'/train.fr','w') as fs:
      with open(raw_dir+'/train.en','w') as ft:
        for i in sel_indices:
          fs.write(bag_srcs[i])
          ft.write(bag_tgts[i])'''
    
    train_log = open('log/train/{}.txt'.format(tmp_dir), 'w')
    subprocess.call('bash src/train_nmt_one_epoch.sh {}/train data-bin-iwslt17/{} data-bin-iwslt17/{} ./checkpoint_best.pt'.format(raw_dir,tmp_dir,tmp_dir),shell=True,stdout=train_log)
    
    val_bleu = 0
    with open('./log/train/E{}S{}.txt'.format(self.cur_epoch,self.cur_step),'r') as f:
      lines = f.readlines()
      if 'nll_loss' in lines[-5].split():
        val_bleu = float(lines[-5].split()[-1])
      elif 'nll_loss' in lines[-6].split():
        val_bleu = float(lines[-6].split()[-1])
    print('\tE{}S{}:  val_bleu {}'.format(self.cur_epoch,self.cur_step,val_bleu))
    if val_bleu > self.best_bleu:
      self.best_bleu = val_bleu
      self.best_epoch = self.cur_epoch
      self.best_timestep = self.cur_step
    
    self.nlp_model = BaseFairseqModel.from_pretrained(model_name_or_path='data-bin-iwslt17/{}'.format(tmp_dir),checkpoint_file='checkpoint_best.pt')
    return val_bleu

  def reset(self):
    # Reset the state of the environment to an initial state
    self.cur_step = 0
    self.shuffled_indices = np.arange(self.num_instances)
    np.random.shuffle(self.shuffled_indices)
    self.reviews_entropy = self.reviews_entropy[self.shuffled_indices]
    self.bag_ids = self.prepare_directory(self.cur_epoch,0,self.shuffled_indices)
    self.cur_bag_emb = self.get_feature(self.nlp_model, self.bag_ids)
    return self.cur_bag_emb


  def render(self, mode='human', close=False):
    # Render the environment to the screen
    pass

  def get_feature(self, fairseq_model, sentence_ids):
    sents_emb = []
    for id in sentence_ids:
      sent = self.text_x_src[id]
      toks = fairseq_model.encode(sent)
      toks = toks.view(1, -1)
      out = fairseq_model.models[0].encoder(toks, len(toks))
      sents_emb.append(torch.mean(out.encoder_out.squeeze(),dim=0).detach().numpy())
    sents_emb = np.array(sents_emb)
    return sents_emb
    

  def prepare_directory(self, episode_count, timestep, shuffled_indices):
    #create directory for current timestep
    save_dir = 'data-bin-iwslt17/E{}S{}'.format(episode_count,timestep)
    if not os.path.isdir(save_dir):
      subprocess.Popen(['mkdir',save_dir])
  
    #remove previous step's directory
    '''if os.path.isdir('E{}S{}'.format(episode_count,timestep-1)):
      subprocess.Popen(['rm','-r','E{}S{}'.format(episode_count,timestep-1)])
    if os.path.isdir('E{}S{}'.format(episode_count-1,max_steps_per_episode-1)):
    subprocess.Popen(['rm','-r','E{}S{}'.format(episode_count-1,max_steps_per_episode-1)])'''
  

    #save ids of cur bag of sentences
    start_idx = timestep * self.bag_size
    end_idx = start_idx + self.bag_size
    sentence_ids = np.array(shuffled_indices[start_idx:end_idx])
    

    #copy valid data to dir
    subprocess.Popen(['cp',os.path.join(self.params.pretrain_data_bin,'dict.fr.txt'),save_dir])
    subprocess.Popen(['cp',os.path.join(self.params.pretrain_data_bin,'dict.en.txt'),save_dir])
    subprocess.Popen(['cp',os.path.join(self.params.pretrain_data_bin,'valid.fr-en.en.bin'),save_dir])
    subprocess.Popen(['cp',os.path.join(self.params.pretrain_data_bin,'valid.fr-en.en.idx'),save_dir])
    subprocess.Popen(['cp',os.path.join(self.params.pretrain_data_bin,'valid.fr-en.fr.bin'),save_dir])
    subprocess.Popen(['cp',os.path.join(self.params.pretrain_data_bin,'valid.fr-en.fr.idx'),save_dir])
    return sentence_ids

class NEAC_Env_Fairseq_Conversation(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, params, nlp_model, text_x_src, text_x_tgt):
    super(NEAC_Env_Fairseq_NMT, self).__init__()
    self.params = params
    self.dump_path = params.dump_path
    self.nlp_model = nlp_model

    self.bag_size = params.bag_size
    self.bag_ids = None
    self.num_instances = len(text_x_tgt)
    self.text_x_src = np.array(text_x_src)
    self.text_x_tgt = np.array(text_x_tgt)
    self.shuffled_indices = np.arange(self.num_instances)

    self.reward_mode = params.reward_mode
    self.cur_step = 0
    self.cur_epoch = 0
    self.best_epoch = 0
    self.best_timestep = 0
    self.best_bleu = 0
    self.num_steps = self.num_instances//self.bag_size
    
    self.reviews_entropy = np.array(compute_unigram_entropy(text_x_src))


    self.action_space=spaces.MultiBinary(self.bag_size)
    self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=
                    (self.bag_size, self.params.fairseq_hidden_dim), dtype=np.float32)
    
    

  def step(self, action):
    # Execute one time step within the environment
    reward = self._get_reward(action)
    val_score = self._take_action(action)
    nxt = self._next_state()
    done = self.cur_step == self.num_steps
    if done:
      self.cur_epoch += 1
    return nxt, reward, done, {'val_score':val_score}

  def _next_state(self):
    self.cur_step += 1
    nxt_bag_ids = self.prepare_directory(self.cur_epoch, self.cur_step, self.shuffled_indices)
    self.bag_ids = nxt_bag_ids
    self.cur_bag_emb = self.get_feature(self.nlp_model, self.bag_ids)
    return self.cur_bag_emb

  def _get_reward(self, action):
    sel_indices = [i for i in range(len(action)) if action[i] == 1]
    if self.reward_mode == 'normalized':
      sum_entropy = sum(self.reviews_entropy[self.bag_ids[sel_indices]])
      return sum_entropy/len(sel_indices)
    elif self.reward_mode == 'ngram_entropy':
      sum_entropy = sum(self.reviews_entropy[self.bag_ids[sel_indices]])
      return sum_entropy
    elif self.reward_mode == 'dispersion':
      return get_dispersion(self.cur_bag_emb[sel_indices])
    elif self.reward_mode == 'convex_hull_volume':
      return get_convex_hull_volume(self.cur_bag_emb[sel_indices])
    elif self.reward_mode == 'graph_entropy':
      return get_graph_entropy(self.cur_bag_emb[sel_indices])
    elif self.reward_mode == 'ensemble':
      ge = get_graph_entropy(self.cur_bag_emb[sel_indices])
      ensemble = ge + 0.1 * sum_entropy
      return ensemble
    
    
      
  
  def _take_action(self, action):
    sel_indices = [i for i in range(len(action)) if action[i] == 1]

    tmp_dir = 'E{}S{}'.format(self.cur_epoch,self.cur_step)
    raw_dir = 'raw/'+tmp_dir
    if not os.path.isdir(raw_dir):
      subprocess.Popen(['mkdir',raw_dir])
    bag_srcs = self.text_x_src[self.bag_ids]
    bag_tgts = self.text_x_tgt[self.bag_ids]
    with open(raw_dir+'/train.src','w') as fs:
      with open(raw_dir+'/train.tgt','w') as ft:
        for i in sel_indices:
          fs.write(bag_srcs[i])
          ft.write(bag_tgts[i])
    
    train_log = open('log/train/{}.txt'.format(tmp_dir), 'w')
    subprocess.call('bash src/train_conversation_one_epoch.sh {}/train data-bin-emp/{} data-bin-emp/{} ./checkpoint_best.pt'.format(raw_dir,tmp_dir,tmp_dir),shell=True,stdout=train_log)
    
    """TO BE DONE
    """
    val_loss = 0
    with open('./log/train/E{}S{}.txt'.format(self.cur_epoch,i),'r') as f:
      lines = f.readlines()
      if 'nll_loss' in lines[-5].split():
        val_bleu = float(lines[-5].split()[-1])
      elif 'nll_loss' in lines[-6].split():
        val_bleu = float(lines[-6].split()[-1])
    print('\tE{}S{}:  val_bleu {}'.format(self.cur_epoch,self.cur_step,val_bleu))
    if val_bleu > self.best_bleu:
      self.best_bleu = val_bleu
      self.best_epoch = self.cur_epoch
      self.best_timestep = self.cur_step
    
    self.nlp_model = BARTModel.from_pretrained(model_name_or_path='data-bin-emp/{}'.format(tmp_dir),checkpoint_file='checkpoint_best.pt')
    return val_bleu

  def reset(self):
    # Reset the state of the environment to an initial state
    self.cur_step = 0
    self.shuffled_indices = np.arange(self.num_instances)
    np.random.shuffle(self.shuffled_indices)
    self.reviews_entropy = self.reviews_entropy[self.shuffled_indices]
    self.bag_ids = self.prepare_directory(self.cur_epoch,0,self.shuffled_indices)
    self.cur_bag_emb = self.get_feature(self.nlp_model, self.bag_ids)
    return self.cur_bag_emb


  def render(self, mode='human', close=False):
    # Render the environment to the screen
    pass
  
  def get_feature(self, fairseq_model, sentence_ids):
    sents_emb = []
    for id in sentence_ids:
      sent = self.text_x_src[id]
      toks = fairseq_model.encode(sent)
      all_layers = fairseq_model.extract_features(toks, return_all_hiddens=True)
      sents_emb.append(torch.mean(all_layers[0].squeeze(),dim=0).detach().numpy())
    sents_emb = np.array(sents_emb)
    return sents_emb
    

  def prepare_directory(self, episode_count, timestep, shuffled_indices):
    #create directory for current timestep
    save_dir = 'data-bin-emp/E{}S{}'.format(episode_count,timestep)
    if not os.path.isdir(save_dir):
      subprocess.Popen(['mkdir',save_dir])
  
    #remove previous step's directory
    '''if os.path.isdir('E{}S{}'.format(episode_count,timestep-1)):
      subprocess.Popen(['rm','-r','E{}S{}'.format(episode_count,timestep-1)])
    if os.path.isdir('E{}S{}'.format(episode_count-1,max_steps_per_episode-1)):
    subprocess.Popen(['rm','-r','E{}S{}'.format(episode_count-1,max_steps_per_episode-1)])'''
  

    #save ids of cur bag of sentences
    start_idx = timestep * self.bag_size
    end_idx = start_idx + self.bag_size
    sentence_ids = np.array(shuffled_indices[start_idx:end_idx])
    

    #copy valid data to dir
    subprocess.Popen(['cp',os.path.join(self.params.pretrain_data_bin,'dict.txt'),save_dir])
    subprocess.Popen(['cp',os.path.join(self.params.pretrain_data_bin,'valid.src-tgt.src.bin'),save_dir])
    subprocess.Popen(['cp',os.path.join(self.params.pretrain_data_bin,'valid.src-tgt.src.idx'),save_dir])
    subprocess.Popen(['cp',os.path.join(self.params.pretrain_data_bin,'valid.src-tgt.tgt.bin'),save_dir])
    subprocess.Popen(['cp',os.path.join(self.params.pretrain_data_bin,'valid.src-tgt.tgt.idx'),save_dir])
    return sentence_ids

