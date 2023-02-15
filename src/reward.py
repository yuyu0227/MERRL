
import numpy as np
from torchtext.data import get_tokenizer
from collections import Counter, OrderedDict
from nltk.util import ngrams



def entropy(prob, symbols):
    p = np.array([prob[w] for w in symbols])
    logp = np.log2(p)
    return -float(np.dot(p, logp))
def renyi_entropy(alpha, prob, symbols):
    p = np.array([prob[w] for w in symbols])
    summed = sum([np.power(px, alpha) for px in p])
    if summed == 0:
        # 0 if none of the words appear in the dictionary;
        # set to a small constant == low prob instead
        summed = 0.0001
    score = 1 / (1 - alpha) * np.log2(summed)
    return score

def min_entropy(prob, symbols):
  p = np.array([prob[w] for w in symbols])
  return -np.log(max(p))


def compute_unigram_entropy(raw_sentence_list):
    tokenizer = get_tokenizer("basic_english")
    total_1gram_list, total_2gram_list, total_3gram_list, total_4gram_list, total_5gram_list = [],[],[],[],[]
    for sent in raw_sentence_list:
      if isinstance(sent, str):
        sent = tokenizer(sent)
      if len(sent) >= 1:
        total_1gram_list.extend(sent)
      if len(sent) >= 2:
        total_2gram_list.extend(list(ngrams(sent,2)))
      if len(sent) >= 3:
        total_3gram_list.extend(list(ngrams(sent,3)))
      if len(sent) >= 4:
        total_4gram_list.extend(list(ngrams(sent,4)))
      if len(sent) >= 5:
        total_5gram_list.extend(list(ngrams(sent,5)))
    counter1, counter2, counter3, counter4, counter5 = Counter(total_1gram_list), \
    Counter(total_2gram_list), Counter(total_3gram_list), Counter(total_4gram_list), Counter(total_5gram_list)
    
    total_freq1,total_freq2,total_freq3,total_freq4,total_freq5 = sum(list(counter1.values())),\
    sum(list(counter2.values())), sum(list(counter3.values())), sum(list(counter4.values())), sum(list(counter5.values()))
    
    rela_freq1, rela_freq2, rela_freq3, rela_freq4, rela_freq5, = {k:v/total_freq1 for k,v in counter1.items()}, \
    {k:v/total_freq2 for k,v in counter2.items()}, {k:v/total_freq3 for k,v in counter3.items()},\
    {k:v/total_freq4 for k,v in counter4.items()}, {k:v/total_freq5 for k,v in counter5.items()}

    min_entropy_per_sentence = [0] * len(raw_sentence_list)
    renyi_entropy_per_sentence = [0] * len(raw_sentence_list)
    shannon_entropy_per_sentence = [0] * len(raw_sentence_list)
    for i in range(len(raw_sentence_list)):
      if isinstance(raw_sentence_list[0], list):
        word_list = raw_sentence_list[i]
      else:
        word_list = tokenizer(raw_sentence_list[i])
      if len(word_list) == 0:
        shannon_entropy_per_sentence[i] = 0
        renyi_entropy_per_sentence[i] = 0
        min_entropy_per_sentence[i] = 0
        continue
      
      min_entropy_list = []
      if len(word_list) >= 1:
        min_entropy_list.append(min_entropy(rela_freq1, word_list))
      if len(word_list) >= 2:
        min_entropy_list.append(min_entropy(rela_freq2, list(ngrams(word_list, 2))))

      renyi_entropy_list = []
      if len(word_list) >= 1:
        renyi_entropy_list.append(renyi_entropy(5, rela_freq1, word_list))
      
      shannon_entropy_list = []
      if len(word_list) >= 1:
        shannon_entropy_list.append(entropy(rela_freq1, word_list))
      if len(word_list) >= 2:
        shannon_entropy_list.append(entropy(rela_freq2, list(ngrams(word_list, 2))))
      '''if len(word_list) >= 3:
        entropy_list.append(entropy(rela_freq3, list(ngrams(word_list, 3))))
      if len(word_list) >= 4:
        entropy_list.append(entropy(rela_freq4, list(ngrams(word_list, 4))))
      if len(word_list) >= 5:
        entropy_list.append(entropy(rela_freq5, list(ngrams(word_list, 5))))'''
      renyi_entropy_per_sentence[i] = np.mean(renyi_entropy_list)
      shannon_entropy_per_sentence[i] = np.mean(shannon_entropy_list)
      min_entropy_per_sentence[i] = np.mean(min_entropy_list)
    
    return shannon_entropy_per_sentence, renyi_entropy_per_sentence, min_entropy_per_sentence

def lines_to_ngrams(lines, n=3):
    ngrams = []
    for s in lines:
      if isinstance(s, str):
        words = [e for e in s.replace('.','').replace('\n','').split(' ') if e != '']
      else:
        words = s
      ngrams.append([tuple(words[i:i + n]) for i in range(len(words) - n + 1)])
    return ngrams
def normalized_unique_ngrams( ngram_lists):
    #Calc the portion of unique n-grams out of all n-grams.
    ngrams = [item for sublist in ngram_lists for item in sublist]  # flatten
    return len(set(ngrams)) / len(ngrams) if len(ngrams) > 0 else 0.
  
def compute_distinct_ngrams( raw_sentence_list, n=1):
    distinct_ngrams = []
    if len(raw_sentence_list) >= 1:
      distinct_ngrams.append(normalized_unique_ngrams(lines_to_ngrams(raw_sentence_list, n=1)))
    if len(raw_sentence_list) >= 2:
      distinct_ngrams.append(normalized_unique_ngrams(lines_to_ngrams(raw_sentence_list, n=2)))
    if len(raw_sentence_list) >= 3:
      distinct_ngrams.append(normalized_unique_ngrams(lines_to_ngrams(raw_sentence_list, n=3)))
    if len(raw_sentence_list) >= 4:
      distinct_ngrams.append(normalized_unique_ngrams(lines_to_ngrams(raw_sentence_list, n=4)))
    if len(raw_sentence_list) >= 5:
      distinct_ngrams.append(normalized_unique_ngrams(lines_to_ngrams(raw_sentence_list, n=5)))
    return np.mean(distinct_ngrams)
