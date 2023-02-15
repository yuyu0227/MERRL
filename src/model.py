import torch
import torch.nn as nn
import numpy as np
import gym
from transformers import AutoConfig,BartTokenizer, BartForConditionalGeneration
from transformers import AutoModelWithLMHead
from transformers import BertModel
from src.dataloader import all_labels, pad_token_label_id
from src.conll2002_measure import *
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
#from sentence_transformers import SentenceTransformer

class BertTagger(nn.Module):
    def __init__(self, params):
        super().__init__()

        
        self.num_tag = len(all_labels)
        self.hidden_dim = params.bert_hidden_dim
        config = AutoConfig.from_pretrained("bert-base-cased")
        config.output_hidden_states = True
        self.model = BertModel.from_pretrained("bert-base-cased",config=config)
        #self.dropout = nn.Dropout(params.dropout)
        self.linear = nn.Linear(self.hidden_dim, self.num_tag)
        
    def forward(self, X):
        outputs = self.model(X) 
        pooled_output = outputs[0]
        prediction = self.linear(pooled_output)
        return prediction

    def get_feature(self, X):
        outputs = self.model(X) # a tuple ((bsz,seq_len,hidden_dim), (bsz, hidden_dim))
        return torch.mean(outputs[0],dim=1)




class BertClassifier(nn.Module):
    def __init__(self, params):
        super().__init__()

        
        self.num_tag = 6
        self.hidden_dim = params.bert_hidden_dim
        config = AutoConfig.from_pretrained("bert-base-cased")
        config.output_hidden_states = True
        self.model = BertModel.from_pretrained("bert-base-cased",config=config)
        #self.dropout = nn.Dropout(params.dropout)
        self.linear1 = nn.Linear(768, 100)
        self.linear2 = nn.Linear(100, 10)
        self.linear3 = nn.Linear(10, self.num_tag)
        
        
    def forward(self, X):
        outputs = self.model(X) 
        pooled_output = outputs[1]

        x1 = self.linear1(pooled_output)
        x2 = self.linear2(x1)
        logits = self.linear3(x2)
        return logits.view(-1, self.num_tag)


class CNN(nn.Module):
    def __init__(self, params, vocab ):
        super(CNN, self).__init__()
        self.emb_file = params.emb_file
        self.embedding_dim = params.cnn_emb_dim
        self.n_filters = params.n_filters
        self.filter_sizes = params.filter_sizes
        self.output_dim = 2 #positive/ negative
        self.dropout_rate = params.dropout

        if self.emb_file != "":
            self.pretrained_embedding = torch.tensor(load_pretrained_vectors(vocab.get_stoi(), self.emb_file))
            self.vocab_size, self.embed_dim = self.pretrained_embedding.shape
            self.embedding = nn.Embedding.from_pretrained(self.pretrained_embedding,
                                                          freeze=False)
        else:
            self.embedding = nn.Embedding(len(vocab), self.embedding_dim)

        
        self.convs = nn.ModuleList([nn.Conv1d(self.embedding_dim, 
                                              self.n_filters, 
                                              filter_size) 
                                    for filter_size in self.filter_sizes])
        self.fc = nn.Linear(len(self.filter_sizes) * self.n_filters, self.output_dim)
        self.dropout = nn.Dropout(self.dropout_rate)
        
        
    def forward(self, ids):
        # ids = [batch size, seq len]
        embedded = self.embedding(ids)
        # embedded = [batch size, seq len, embedding dim]
        embedded = embedded.permute(0,2,1)
        # embedded = [batch size, embedding dim, seq len]
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n filters, seq len - filter_sizes[n] + 1]
        pooled = [conv.max(dim=-1).values for conv in conved]
        # pooled_n = [batch size, n filters]
        cat = self.dropout(torch.cat(pooled, dim=-1))
        # cat = [batch size, n filters * len(filter_sizes)]
        prediction = self.fc(cat)
        # prediction = [batch size, output dim]
        return prediction
    
    def get_feature(self, ids):
        # ids = [batch size, seq len]
        embedded = self.embedding(ids)
        # embedded = [batch size, seq len, embedding dim]
        embedded = embedded.permute(0,2,1)
        # embedded = [batch size, embedding dim, seq len]
        conved = [torch.relu(conv(embedded)) for conv in self.convs]
        # conved_n = [batch size, n filters, seq len - filter_sizes[n] + 1]
        pooled = [conv.max(dim=-1).values for conv in conved]
        # pooled_n = [batch size, n filters]
        cat = torch.cat(pooled, dim=-1)
        # cat = [batch size, n filters * len(filter_sizes)]
        return cat


class CustomEncoder(BaseFeaturesExtractor):
    """
    :param observation_space: (gym.Space)
    :param features_dim: (int) Number of features extracted.
        This corresponds to the number of unit for the last layer.
    """

    def __init__(self, observation_space: gym.spaces.Box, features_dim: int = 384):
        super(CustomEncoder, self).__init__(observation_space, features_dim)
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return self.encoder.forward(observations)['sentence_embedding']



class BartConversationModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.tokenizer = BartTokenizer.from_pretrained('facebook/bart-base')
        self.model = BartForConditionalGeneration.from_pretrained('facebook/bart-base')
    
    def forward(self, input_ids, attention_mask, decoder_input_ids):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            decoder_input_ids=decoder_input_ids
        )
        
    def get_feature(self, input_ids, attention_mask):
        encoder = self.model.get_encoder()
        sequence_output = encoder.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )[0]
        return torch.mean(sequence_output,dim=1)