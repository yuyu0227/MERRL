import gym
import os
from gym import spaces
import numpy as np
import torch
#from sentence_transformers import SentenceTransformer
from torch.utils.data import DataLoader
from src.dataloader import Dataset, collate_fn, collate_fn_for_sentiment, DialogueBatchDataset
from src.reward import *


class MESS_Sequential_Env(gym.Env):
  """Custom Environment that follows gym interface"""
  metadata = {'render.modes': ['human']}

  def __init__(self, params, trainer, train_dataloader, text_x_train, dev_dataloader=None):
    super(MESS_Sequential_Env, self).__init__()

    self.val_score_record = []
    self.train_score_record = []

    '''
    try to save init model
    '''
    #self.init_trainer = trainer


    self.params = params
    self.dump_path = params.dump_path

    self.nlp_trainer = trainer
    self.trainloader = train_dataloader
    self.devloader = dev_dataloader
    self.best_val_score = 0

    self.bag_size = params.bag_size
    self.batch_size = params.batch_size
    self.task = params.task
    self.num_instances = len(text_x_train)

    

    self.text_x_train = np.array(text_x_train)
    self.bag_indices = None
    self.bag_features = None
    self.bag_labels = None
    self.cur_batch_emb = None

    self.reward_mode = params.reward_mode
    self.epoch = 0
    self.cur_step = 0
    self.num_steps = self.num_instances//self.bag_size
    self.batch_start = 0
    self.shannon_entropy, self.renyi_entropy, self.min_entropy = np.array(compute_unigram_entropy(text_x_train))

    self.action_space=spaces.Box(0, 1, (self.bag_size,), dtype=np.int8)
    #self.action_space=spaces.MultiBinary(self.bag_size)
   
    if params.task == 'ner':
      self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=
                    (self.bag_size, self.params.bert_hidden_dim), dtype=np.float32)
    elif params.task == 'sentiment':
      self.observation_space = spaces.Box(low=-float("inf"), high=float("inf"), shape=
                    (self.bag_size, self.params.cnn_emb_dim), dtype=np.float32)
    

  def step(self, action):
    # Execute one time step within the environment
    if self.reward_mode == 'pred_entropy':
        scores, reward = self._take_action(action)
    else:
        reward = self._get_reward(action)
        scores = self._take_action(action)
    
    train_score = scores[0]
    val_score = scores[1]
    nxt = self._next_state()
    self.cur_step += 1
    done = self.cur_step == self.num_steps
    if done:
      self.epoch += 1
      self.val_score_record.append(val_score)
      self.train_score_record.append(train_score)
      print(f'\t Epoch: {self.epoch}   |  Val. Acc: {val_score*100:.2f}% ')
    return nxt, reward, done, {'val_score':val_score}

  def _next_state(self):
    self.bag_indices, self.bag_features, self.bag_labels = next(iter(self.trainloader))
    if self.params.task == 'sentiment':
      cur_loader = DataLoader(dataset=Dataset(self.bag_indices, self.bag_features, self.bag_labels), batch_size=self.bag_features.size()[0], shuffle=True, collate_fn=collate_fn_for_sentiment)
    else:
      cur_loader = DataLoader(dataset=Dataset(self.bag_indices, self.bag_features, self.bag_labels), batch_size=self.bag_features.size()[0], shuffle=True, collate_fn=collate_fn)
    _, next_batch, _= next(iter(cur_loader))
    next_batch = next_batch.cuda()
    self.cur_batch_emb = self.nlp_trainer.model.get_feature(next_batch).detach().cpu().numpy()
    return self.cur_batch_emb

  def _get_reward(self, action):
    #print('action',action)
    sel_indices = [i for i in range(len(action)) if action[i] > 0.5]
    if self.reward_mode == 'shannon_entropy':
      sum_entropy = sum(self.shannon_entropy[self.bag_indices[sel_indices]])
      return sum_entropy
    elif self.reward_mode == 'renyi_entropy':
      sum_entropy = sum(self.renyi_entropy[self.bag_indices[sel_indices]])
      return sum_entropy
    elif self.reward_mode == 'min_entropy':
      sum_entropy = sum(self.min_entropy[self.bag_indices[sel_indices]])
      return sum_entropy
    
    
      

  def _take_action(self, action):
    #print('action',action)
    sel_indices = [int(i) for i in range(len(action)) if action[i] > 0.5]
    #print('sel_indices',sel_indices)
    if self.task == 'sentiment':
      batch_dataloader = DataLoader(dataset=Dataset(self.bag_indices[sel_indices], self.bag_features[sel_indices], self.bag_labels[sel_indices]), batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn_for_sentiment)
    elif self.task == 'ner':
      batch_dataloader = DataLoader(dataset=Dataset(self.bag_indices[sel_indices], self.bag_features[sel_indices], self.bag_labels[sel_indices]), batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
    if self.reward_mode == 'pred_entropy':
        train_score, val_score, y_hat_entropy = self.nlp_trainer.train(batch_dataloader, self.devloader, epoch=self.params.inner_epoch)
    else:
        train_score, val_score = self.nlp_trainer.train(batch_dataloader, self.devloader, epoch=self.params.inner_epoch)
    
    if val_score > self.best_val_score:
      self.best_val_score = val_score
      #print('Found new best model...')
      torch.save(self.nlp_trainer.model.state_dict(), os.path.join(self.dump_path, 'best_finetune_model.pth'))
    
    if self.reward_mode == 'pred_entropy':
        return (train_score, val_score),  -y_hat_entropy
    return (train_score, val_score)
    
    

  def reset(self):
    # Reset the state of the environment to an initial state
    self.cur_step = 0
    self.bag_indices, self.bag_features, self.bag_labels = next(iter(self.trainloader))
    if self.task == 'sentiment':
      cur_loader = DataLoader(dataset=Dataset(self.bag_indices, self.bag_features, self.bag_labels), batch_size=self.bag_features.size()[0], shuffle=True, collate_fn=collate_fn_for_sentiment)
    elif self.task == 'ner':
      cur_loader = DataLoader(dataset=Dataset(self.bag_indices, self.bag_features, self.bag_labels), batch_size=self.bag_features.size()[0], shuffle=True, collate_fn=collate_fn)
    _, first_batch, _= next(iter(cur_loader))
    first_batch = first_batch.cuda()

    '''Technically, the nlp model should be reset to initail state as well??? since it's part of the environment'''
    #self.nlp_trainer = self.init_trainer
    
    self.cur_batch_emb = self.nlp_trainer.model.get_feature(first_batch).detach().cpu().numpy()

    
    return self.cur_batch_emb

  def render(self, mode='human', close=False):
    # Render the environment to the screen
    pass


