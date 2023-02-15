import os
import csv
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchtext.vocab import build_vocab_from_iterator
from torchtext.data.functional import numericalize_tokens_from_iterator
from collections import defaultdict
import numpy as np
from tqdm import tqdm
import random
import logging
logger = logging.getLogger()

# from transformers import BertTokenizer
# bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
from src.config import get_params
params = get_params()
from transformers import AutoTokenizer
auto_tokenizer = AutoTokenizer.from_pretrained(params.model_name)
pad_token_label_id = nn.CrossEntropyLoss().ignore_index

all_amazon_domains = ['apparel','automotive','baby','beauty','camera_&_photo',\
		'cell_phones_&_service','computer_&_video_games','gourmet_food','grocery',\
			'health_&_personal_care','jewelry_&_watches','magazines','music','musical_instruments','office_products',\
				'outdoor_living','software','sports_&_outdoors','tools_&_hardware','toys_&_games',\
					'video']


politics_labels = ['O', 'B-country', 'B-politician', 'I-politician', 'B-election', 'I-election', 'B-person', 'I-person', 'B-organisation', 'I-organisation', 'B-location', 'B-misc', 'I-location', 'I-country', 'I-misc', 'B-politicalparty', 'I-politicalparty', 'B-event', 'I-event']
science_labels = ['O', 'B-scientist', 'I-scientist', 'B-person', 'I-person', 'B-university', 'I-university', 'B-organisation', 'I-organisation', 'B-country', 'I-country', 'B-location', 'I-location', 'B-discipline', 'I-discipline', 'B-enzyme', 'I-enzyme', 'B-protein', 'I-protein', 'B-chemicalelement', 'I-chemicalelement', 'B-chemicalcompound', 'I-chemicalcompound', 'B-astronomicalobject', 'I-astronomicalobject', 'B-academicjournal', 'I-academicjournal', 'B-event', 'I-event', 'B-theory', 'I-theory', 'B-award', 'I-award', 'B-misc', 'I-misc']
music_labels = ['O', 'B-musicgenre', 'I-musicgenre', 'B-song', 'I-song', 'B-band', 'I-band', 'B-album', 'I-album', 'B-musicalartist', 'I-musicalartist', 'B-musicalinstrument', 'I-musicalinstrument', 'B-award', 'I-award', 'B-event', 'I-event', 'B-country', 'I-country', 'B-location', 'I-location', 'B-organisation', 'I-organisation', 'B-person', 'I-person', 'B-misc', 'I-misc']
literature_labels = ["O", "B-book", "I-book", "B-writer", "I-writer", "B-award", "I-award", "B-poem", "I-poem", "B-event", "I-event", "B-magazine", "I-magazine", "B-literarygenre", "I-literarygenre", 'B-country', 'I-country', "B-person", "I-person", "B-location", "I-location", 'B-organisation', 'I-organisation', 'B-misc', 'I-misc']
ai_labels = ["O", "B-field", "I-field", "B-task", "I-task", "B-product", "I-product", "B-algorithm", "I-algorithm", "B-researcher", "I-researcher", "B-metrics", "I-metrics", "B-programlang", "I-programlang", "B-conference", "I-conference", "B-university", "I-university", "B-country", "I-country", "B-person", "I-person", "B-organisation", "I-organisation", "B-location", "I-location", "B-misc", "I-misc"]

domain2labels = {"politics": politics_labels, "science": science_labels, "music": music_labels, "literature": literature_labels, "ai": ai_labels}

all_labels = set()
all_labels = all_labels.union(politics_labels)
all_labels = all_labels.union(science_labels)
all_labels = all_labels.union(music_labels)
all_labels = all_labels.union(literature_labels)
all_labels = all_labels.union(ai_labels)
all_labels = list(all_labels)

def read_ner(datapath):
	raw_texts, inputs, labels = [], [], []
	with open(datapath, "r") as fr:
		raw_token_list, token_list, label_list = [], [], []
		for i, line in enumerate(fr):
			line = line.strip()
			if line == "":
				if len(token_list) > 0:
					assert len(token_list) == len(label_list)
					inputs.append([auto_tokenizer.cls_token_id] + token_list + [auto_tokenizer.sep_token_id])
					labels.append([pad_token_label_id] + label_list + [pad_token_label_id])
					raw_texts.append(raw_token_list)
				raw_token_list, token_list, label_list = [], [], []
				continue
			
			splits = line.split("\t")
			token = splits[0]
			raw_token_list.append(token)
			
			label = splits[1]
			

			subs_ = auto_tokenizer.tokenize(token)
			if len(subs_) > 0:
				label_list.extend([all_labels.index(label)] + [pad_token_label_id] * (len(subs_) - 1))
				token_list.extend(auto_tokenizer.convert_tokens_to_ids(subs_))
			else:
				print("length of subwords for %s is zero; its label is %s" % (token, label))

	return raw_texts, inputs, labels






class Dataset(data.Dataset):
	def __init__(self, indices, inputs, labels):
		self.indices = indices
		self.X = inputs
		self.y = labels
	
	def __getitem__(self, index):
		return self.indices[index], self.X[index], self.y[index]

	def __len__(self):
		return len(self.X)


def read_lm(datapath):
	inputs= []
	with open(datapath,'r') as fr:
		lines = fr.readlines()
	cleaned_lines = []
	for l in lines:
		l = l.strip()
		if len(l) > 0:
			cleaned_lines.append(l)
	for l in cleaned_lines:
		tokens = auto_tokenizer.tokenize(l)
		inputs.append(auto_tokenizer.convert_tokens_to_ids(tokens))
	return inputs
def get_lm_domain_classification_dataloader(batch_size):
	inputs_wiki = read_lm()
	labels_wiki = np.zeros(shape=(len(inputs_wiki)))
	inputs_wiki_dev = read_lm()
	labels_wiki_dev = np.zeros(shape=(len(inputs_wiki_dev)))

	inputs_ptb = read_lm()
	labels_ptb = np.ones(shape=(len(inputs_ptb)))
	inputs_ptb_dev = read_lm()
	labels_ptb_dev = np.ones(shape=(len(inputs_ptb_dev)))

	inputs_iwslt17 = read_lm()
	labels_iwslt17 = np.full(shape=len(inputs_iwslt17), fill_value=2)
	inputs_bio21 = read_lm()
	labels_bio21 = np.full(shape=len(inputs_bio21), fill_value=3)

	inputs_train, labels_train, inputs_dev, labels_dev = [],[],[],[]
	inputs_train.extend(inputs_wiki)
	inputs_train.extend(inputs_ptb)
	inputs_train.extend(inputs_iwslt17)
	inputs_train.extend(inputs_bio21)

	labels_train.extend(labels_wiki)
	labels_train.extend(labels_ptb)
	labels_train.extend(labels_iwslt17)
	labels_train.extend(labels_bio21)

	inputs_dev.extend(inputs_wiki_dev)
	inputs_dev.extend(inputs_ptb_dev)
	labels_dev.extend(labels_wiki_dev)
	labels_dev.extend(labels_ptb_dev)
	
	dataset_train = Dataset(np.arange(len(inputs_train)), inputs_train, labels_train)
	dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_sentiment)

	dataset_dev = Dataset(np.arange(len(inputs_dev)), inputs_dev, labels_dev)
	dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_sentiment)
	return dataloader_train, dataloader_dev

'''NER data'''


def collate_fn(data):
	indices, X, y = zip(*data)
	lengths = [len(bs_x) for bs_x in X]
	max_lengths = max(lengths)
	padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(auto_tokenizer.pad_token_id)
	padded_y = torch.LongTensor(len(X), max_lengths).fill_(pad_token_label_id)
	for i, (seq, y_) in enumerate(zip(X, y)):
		length = lengths[i]
		padded_seqs[i, :length] = torch.LongTensor(seq)
		padded_y[i, :length] = torch.LongTensor(y_)

	return np.array(indices), padded_seqs, padded_y



def get_ner_domain_classification_dataloader(batch_size):
	raw_texts, inputs_train, labels_train = read_ner("data/ner_data/conll2003/train.txt")
	_, inputs_dev, labels_dev = read_ner("data/ner_data/conll2003/dev.txt")
	labels_conll = np.zeros(shape=(len(inputs_train)))
	labels_conll_dev = np.zeros(shape=(len(inputs_dev)))
	
	_, pol_train, _ = read_ner("data/ner_data/politics/train.txt")
	_, pol_dev, _ = read_ner("data/ner_data/politics/dev.txt")
	labels_pol = np.ones(shape=(len(pol_train)))
	labels_pol_dev = np.ones(shape=(len(pol_dev)))

	_, sci_train, _ = read_ner("data/ner_data/science/train.txt")
	_, sci_dev, _ = read_ner("data/ner_data/science/dev.txt")
	labels_sci = np.full(shape=len(sci_train), fill_value=2)
	labels_sci_dev = np.full(shape=len(sci_dev), fill_value=2)

	_, music_train, _ = read_ner("data/ner_data/music/train.txt")
	_, music_dev, _ = read_ner("data/ner_data/music/dev.txt")
	labels_music = np.full(shape=len(music_train), fill_value=3)
	labels_music_dev = np.full(shape=len(music_dev), fill_value=3)


	_, lit_train, _ = read_ner("data/ner_data/literature/train.txt")
	_, lit_dev, _ = read_ner("data/ner_data/literature/dev.txt")
	labels_lit = np.full(shape=len(lit_train), fill_value=4)
	labels_lit_dev = np.full(shape=len(lit_dev), fill_value=4)
	
	_, ai_train, _ = read_ner("data/ner_data/ai/train.txt")
	_, ai_dev, _ = read_ner("data/ner_data/ai/dev.txt")
	labels_ai = np.full(shape=len(ai_train), fill_value=5)
	labels_ai_dev = np.full(shape=len(ai_dev), fill_value=5)
	inputs_train.extend(pol_train)
	inputs_train.extend(sci_train)
	inputs_train.extend(music_train)
	inputs_train.extend(lit_train)
	inputs_train.extend(ai_train)
	inputs_dev.extend(pol_dev)
	inputs_dev.extend(sci_dev)
	inputs_dev.extend(music_dev)
	inputs_dev.extend(lit_dev)
	inputs_dev.extend(ai_dev)
	labels_train = np.concatenate((labels_conll, labels_pol, labels_sci, labels_music, labels_lit, labels_ai))
	labels_dev = np.concatenate((labels_conll_dev, labels_pol_dev, labels_sci_dev, labels_music_dev, labels_lit_dev, labels_ai_dev))

	dataset_train = Dataset(np.arange(len(inputs_train)), inputs_train, labels_train)
	dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_sentiment)

	dataset_dev = Dataset(np.arange(len(inputs_dev)), inputs_dev, labels_dev)
	dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_sentiment)
	return dataloader_train, dataloader_dev
def get_ner_target_dataloader(tgt_dm, batch_size):
	
	inputs_target_dev, labels_target_dev = [],[]
	inputs_target_test, labels_target_test = [],[]
	_, inputs_train, labels_train = read_ner("data/ner_data/%s/train.txt" % tgt_dm)
	_, inputs_dev, labels_dev = read_ner("data/ner_data/%s/dev.txt" % tgt_dm)
	_, inputs_test, labels_test = read_ner("data/ner_data/%s/test.txt" % tgt_dm)

	inputs_target_dev.extend(inputs_train)
	inputs_target_test.extend(inputs_dev)
	inputs_target_test.extend(inputs_test)
	labels_target_dev.extend(labels_train)
	labels_target_test.extend(labels_dev)
	labels_target_test.extend(labels_test)

	print("Load {} domain data: {} valid samples, {} test samples".format(tgt_dm, len(inputs_target_dev), len(inputs_target_test)))
	dataset_target_dev = Dataset(np.arange(len(inputs_target_dev)),inputs_target_dev, labels_target_dev)
	dataloader_target_dev = DataLoader(dataset=dataset_target_dev, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	dataset_target_test = Dataset(np.arange(len(inputs_target_test)), inputs_target_test, labels_target_test)
	dataloader_target_test = DataLoader(dataset=dataset_target_test, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	return dataloader_target_dev, dataloader_target_test




def get_conll2003_dataloader(batch_size, bag_size):
	raw_texts, inputs_train, labels_train = read_ner("data/ner_data/conll2003/train.txt")
	_, inputs_dev, labels_dev = read_ner("data/ner_data/conll2003/dev.txt")
	_, inputs_test, labels_test = read_ner("data/ner_data/conll2003/test.txt")

	print("conll2003 dataset: train size: %d; dev size %d; test size: %d" % (len(inputs_train), len(inputs_dev), len(inputs_test)))

	dataset_train = Dataset(np.arange(len(inputs_train)), inputs_train, labels_train)
	dataset_dev = Dataset(np.arange(len(inputs_dev)), inputs_dev, labels_dev)
	dataset_test = Dataset(np.arange(len(inputs_test)), inputs_test, labels_test)
	
	dataloader_pretrain = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn)
	dataloader_train = DataLoader(dataset=dataset_train, batch_size=bag_size, shuffle=True, collate_fn=collate_fn)
	dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)
	dataloader_test = DataLoader(dataset=dataset_test, batch_size=batch_size, shuffle=False, collate_fn=collate_fn)

	return raw_texts, dataloader_pretrain, dataloader_train, dataloader_dev, dataloader_test


'''Sentiment data'''
def get_sentiment_domain_classification_dataloader(batch_size):
	def get_data_and_label(base_dir,domain,label):
		data = {}
		y_map = {
			'positive': 1,
			'negative': 0
		}
		entire_path = base_dir + '/'+domain+'/'+label+'.review'
		with open(entire_path,encoding = "ISO-8859-1") as f:
			d = f.readlines()
			data['reviews'] = []
			data['labels'] = []
			start = False
			for line in d:
				if line == "</review_text>\n":
					start = False
					data['reviews'].append(new_review)
					data['labels'].append(y_map[label])
				if start:
					new_review += line
				if line == "<review_text>\n":
					start = True
					new_review = ""
		return data
	#source, target = load_amazon_processed(base_dir='./data/sentiment/amazon-reviews/processed_acl',target_domain ='electronics')
	source, target = load_amazon_unprocessed(base_dir='./data/sentiment/sorted_data_acl',trg_domain ='electronics')
	#print(len(vocab))
	train_inputs = []
	for l in source['reviews']:
		if len(l) > 512:
			l = l[:512]
		tokens = auto_tokenizer.tokenize(l)
		train_inputs.append(auto_tokenizer.convert_tokens_to_ids(tokens))
	train_labels = list(np.zeros(shape=(len(train_inputs)),dtype=np.int8))

	
	
	
	dev_inputs = []
	for l in target['reviews']:
		if len(l) > 512:
			l = l[:512]
		tokens = auto_tokenizer.tokenize(l)
		dev_inputs.append(auto_tokenizer.convert_tokens_to_ids(tokens))
	
	dev_labels = np.zeros(shape=(len(dev_inputs)))
	all_domains = ['apparel','automotive','baby','beauty','camera_&_photo',\
		'cell_phones_&_service','computer_&_video_games','gourmet_food','grocery',\
			'health_&_personal_care','jewelry_&_watches','magazines','music','musical_instruments','office_products',\
				'outdoor_living','software','sports_&_outdoors','tools_&_hardware','toys_&_games',\
					'video']
	#test_inputs = []
	#test_labels = []
	i = 1
	for domain in all_domains:
		raw_pos = get_data_and_label('./data/sentiment/sorted_data',domain,'positive')
		raw_neg = get_data_and_label('./data/sentiment/sorted_data',domain,'negative')
		reviews = raw_pos['reviews'] + raw_neg['reviews']
		inputs = []
		for l in reviews:
			if len(l) > 512:
				l = l[:512]
			tokens = auto_tokenizer.tokenize(l)
			inputs.append(auto_tokenizer.convert_tokens_to_ids(tokens))
	
		train_labels.extend(list(np.full(shape=len(inputs), fill_value=i,dtype=np.int8)))
		
		#train_labels.extend(list(np.full(shape=len(inputs), fill_value=i)))
		#test_inputs.append(inputs)
		#test_labels.append(np.full(shape=len(inputs), fill_value=i))
		i += 1
		
	
	'''print(len(train_labels))
	lab_set = set()
	for label in train_labels:
		lab_set.add(label)
	print(len(lab_set))'''
	
	dataset_train = Dataset(np.arange(len(train_inputs)), train_inputs, train_labels)
	dataloader_train = DataLoader(dataset=dataset_train, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_sentiment)

	dataset_dev = Dataset(np.arange(len(dev_inputs)), dev_inputs, dev_labels)
	dataloader_dev = DataLoader(dataset=dataset_dev, batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_sentiment)
	return dataloader_train, dataloader_dev



def collate_fn_for_sentiment(data):
	indices, X, y = zip(*data)
	lengths = [len(bs_x) for bs_x in X]
	max_lengths = max(lengths)
	padded_seqs = torch.LongTensor(len(X), max_lengths).fill_(0)
	for i, (seq, y_) in enumerate(zip(X, y)):
		length = lengths[i]
		padded_seqs[i, :length] = torch.LongTensor(seq)

	return np.array(indices), padded_seqs, torch.LongTensor(y) 

def load_amazon_processed(base_dir, target_domain):
	def add_to_list(base_dir, domain, split):
		reviews = []
		with open(base_dir+'/'+domain+'/'+split+'.review','r', encoding='utf-8') as f:
			for line in f:
				features = line.split(' ')
				review = []
				for feat in features:
					ngram, count = feat.split(':')
					if ngram.startswith('#'):
						continue
					for _ in range(int(count)):
						
						review.append(ngram)
				reviews.append(review)
		return reviews

	all_domains = os.listdir(base_dir)
	source = defaultdict(dict)
	target = defaultdict(dict)
	source['reviews'] = []
	target['reviews'] = []
	source['labels'] = []
	target['labels'] = []
  
	for domain in all_domains:
		if domain == '.DS_Store':
			continue
		if domain == target_domain:
			pos = add_to_list(base_dir, target_domain, 'positive')
			target['reviews'].extend(pos)
			target['labels'].extend(np.ones(shape=(len(pos))))
			neg = add_to_list(base_dir, target_domain, 'negative')
			target['reviews'].extend(neg)
			target['labels'].extend(np.zeros(shape=(len(neg))))
		else:
			pos = add_to_list(base_dir, domain, 'positive')
			source['reviews'].extend(pos)
			source['labels'].extend(np.ones(shape=(len(pos))))
			neg = add_to_list(base_dir, domain, 'negative')
			source['reviews'].extend(neg)
			source['labels'].extend(np.zeros(shape=(len(neg))))

	return source, target

def load_amazon_unprocessed(base_dir,trg_domain):
	def get_data_and_label(base_dir,domain,label):
		data = {}
		y_map = {
			'positive': 1,
			'negative': 0
		}
		entire_path = base_dir + '/'+domain+'/'+label+'.review'
		with open(entire_path,encoding = "ISO-8859-1") as f:
			d = f.readlines()
			data['reviews'] = []
			data['labels'] = []
			start = False
			for line in d:
				if line == "</review_text>\n":
					start = False
					data['reviews'].append(new_review)
					data['labels'].append(y_map[label])
				if start:
					new_review += line
				if line == "<review_text>\n":
					start = True
					new_review = ""
		return data
	all_domains = ['dvd','books','kitchen','electronics']
	source = {}
	source['reviews'] = []
	source['labels'] = []
	target = {}
	trg_pos = get_data_and_label(base_dir,trg_domain,'positive')
	trg_neg = get_data_and_label(base_dir,trg_domain,'negative')
	target['reviews'] = trg_pos['reviews'] + trg_neg['reviews']
	target['labels'] = trg_pos['labels'] + trg_neg['labels']
	for domain in all_domains:
		if domain != trg_domain:
			pos = get_data_and_label(base_dir,domain,'positive')
			neg = get_data_and_label(base_dir,domain,'negative')
			source['reviews'] += pos['reviews'] + neg['reviews']
			source['labels'] += pos['labels'] + neg['labels']
	return source, target

def get_sentiment_src_tgt_dataloader(tgt_dm, batch_size, bag_size):

	raw_train_texts = load_amazon_unprocessed(base_dir='./data/sentiment/sorted_data_acl',trg_domain=tgt_dm)[0]['reviews']
	source, target = load_amazon_processed(base_dir='./data/sentiment/amazon-reviews/processed_acl',target_domain =tgt_dm)
	vocab = build_vocab_from_iterator(source['reviews'], min_freq=2)
	vocab.insert_token('<pad>',0)
	vocab.insert_token('<unk>',1)
	vocab.set_default_index(vocab['<unk>'])
	#print(len(vocab))
	train_inputs, train_labels = [], source['labels']
	trainiter = numericalize_tokens_from_iterator(vocab, source['reviews'])
	for seqs in trainiter:
		train_inputs.append([num for num in seqs])

	

	tgt_inputs, tgt_labels = [], target['labels']
	tgtiter = numericalize_tokens_from_iterator(vocab, target['reviews'])
	for seqs in tgtiter:
		tgt_inputs.append([num for num in seqs])
	
	print(f'Succesfully load {len(train_inputs)} source samples, {len(tgt_inputs)} target samples from {tgt_dm} domain.')
	tgt_inputs = np.array(tgt_inputs, dtype=object)
	tgt_labels = np.array(tgt_labels, dtype=object)
	dev_indices = np.random.choice(np.arange(len(tgt_inputs)), len(tgt_inputs)//2)
	test_indices = np.array([i for i in np.arange(len(tgt_inputs)) if i not in dev_indices])
	pretrain_dataloader = DataLoader(dataset=Dataset(np.arange(len(train_inputs)),train_inputs, train_labels), batch_size=batch_size, shuffle=True, collate_fn=collate_fn_for_sentiment)
	train_dataloader = DataLoader(dataset=Dataset(np.arange(len(train_inputs)),train_inputs, train_labels), batch_size=bag_size, shuffle=True, collate_fn=collate_fn_for_sentiment)
	tgt_dev_dataloader = DataLoader(dataset=Dataset(np.arange(len(dev_indices)),tgt_inputs[dev_indices], tgt_labels[dev_indices]), batch_size=batch_size, shuffle=False, collate_fn=collate_fn_for_sentiment)
	tgt_test_dataloader = DataLoader(dataset=Dataset(np.arange(len(test_indices)), tgt_inputs[test_indices], tgt_labels[test_indices]), batch_size=batch_size, shuffle=False, collate_fn=collate_fn_for_sentiment)
	return raw_train_texts, pretrain_dataloader, train_dataloader, tgt_dev_dataloader, tgt_test_dataloader, vocab



def get_sentiment_unseen_dataloaders(base_dir, vocab, batch_size):
	def get_data_and_label(base_dir,domain,label):
		data = {}
		y_map = {
			'positive': 1,
			'negative': 0
		}
		entire_path = base_dir + '/'+domain+'/'+label+'.review'
		with open(entire_path,encoding = "ISO-8859-1") as f:
			d = f.readlines()
			data['reviews'] = []
			data['labels'] = []
			start = False
			for line in d:
				if line == "</review_text>\n":
					start = False
					data['reviews'].append(new_review)
					data['labels'].append(y_map[label])
				if start:
					new_review += line
				if line == "<review_text>\n":
					start = True
					new_review = ""
		return data
	all_domains = ['apparel','automotive','baby','beauty','camera_&_photo',\
		'cell_phones_&_service','computer_&_video_games','gourmet_food','grocery',\
			'health_&_personal_care','jewelry_&_watches','magazines','music','musical_instruments','office_products',\
				'outdoor_living','software','sports_&_outdoors','tools_&_hardware','toys_&_games',\
					'video']
	domains_dataloader = {}
	for domain in all_domains:
		if domain == '.DS_Store':
			continue
		raw = {}
		raw_pos = get_data_and_label(base_dir,domain,'positive')
		raw_neg = get_data_and_label(base_dir,domain,'negative')
		raw['reviews'] = raw_pos['reviews'] + raw_neg['reviews']
		raw['labels'] = raw_pos['labels'] + raw_neg['labels']
		inputs = []
		dataiter = numericalize_tokens_from_iterator(vocab, raw['reviews'])
		for seqs in dataiter:
			inputs.append([num for num in seqs])
			
		domains_dataloader[domain] = DataLoader(Dataset(np.arange(len(inputs)),inputs, raw['labels']), num_workers=1, batch_size=batch_size, 
							  shuffle=True, drop_last=True, collate_fn=collate_fn_for_sentiment)
	return domains_dataloader





'''Conversation'''


class DialogueDataset(Dataset):
    def __init__(self, data_dir, split, tokenizer, max_length):
        src_texts = []
        tgt_texts = []
        with open(os.path.join(data_dir, split + '.tsv')) as f:
            reader = csv.reader(f, delimiter='\t', quoting=csv.QUOTE_NONE)
            for row in reader:
                src_texts.append(row[0])
                tgt_texts.append(row[1])
        
        self.batch = tokenizer.prepare_seq2seq_batch(
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            max_length=max_length,
            return_tensors='pt'
        )

    def __len__(self):
        return self.batch['input_ids'].size(0)

    def __getitem__(self, index):
        input_ids = self.batch['input_ids'][index]
        attention_mask = self.batch['attention_mask'][index]
        labels = self.batch['labels'][index]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
            'index': index,
        }

class DialogueBatchDataset(Dataset):
    def __init__(self, batch):
        self.batch = batch

    def __len__(self):
        return self.batch['input_ids'].size(0)

    def __getitem__(self, index):
        input_ids = self.batch['input_ids'][index]
        attention_mask = self.batch['attention_mask'][index]
        labels = self.batch['labels'][index]

        return {
            'input_ids': input_ids,
            'attention_mask': attention_mask,
            'labels': labels,
        }