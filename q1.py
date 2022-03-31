#!/usr/bin/env python
# coding: utf-8

# In[1]:


import codecs
import numpy as np
from numpy.linalg import norm
import pickle


import gensim
# from gensim.models.word2vec import Word2Vec
from gensim.models import KeyedVectors

# In[2]:


q1, q2=[], []
with codecs.open('Wordsimilarity_datasets/iiith_wordsim/hindi.txt', encoding='utf-8') as f:
    for line in f:
        tokens = line.split('\n')
        if len(tokens[0])>0:
            tokens = tokens[0].split('\t')
            q1.append(tokens[0])
            q2.append(tokens[1])
print("First query vector: ", q1)
print("Second query vector: ", q2)


# In[3]:


def get_cosine_sim_score(v1, v2):
    cos_sim = np.dot(v1, v2)/(norm(v1)*norm(v2))
    return cos_sim

def glove_preprocessing(glove_path):
    glove_vec={}
    with codecs.open(glove_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            tokens = line.split()
            token = tokens[0]
            vec = tokens[1:]
            glove_vec[token] = [float(v) for v in vec]
            print(token, end=' ')
            print(glove_vec[token])

    with open('glove_vec.pickle', 'wb') as handle:
        pickle.dump(glove_vec, handle)
    return glove_vec

def get_glove_similar_words(glove_vec, thresholds, q1, q2):
    similar_word_based_on_thresh = {t:[] for t in thresholds}
    for token1, token2 in zip(q1, q2):
        if token1 in glove_vec.keys():
            print("First Token found: ", token1)
            # print("First Token vec: ", glove_vec[token1])
            # count+=1
            v1 = glove_vec[token1]
        else:
            print("First Token not found: ", token1)
        if token2 in glove_vec.keys():
            print("Second Token found: ", token2)
            # print("Second Token vec: ", glove_vec[token2])
            # count+=1
            v2 = glove_vec[token2]
        else:
            print("Second Token not found: ", token2)
        score = get_cosine_sim_score(v1, v2)
        if score>=0.4:
            for thresh in thresholds:
                similar_word_based_on_thresh[thresh].append((token1, token2))
        elif score>=0.5:
            for thresh in thresholds[1:]:
                similar_word_based_on_thresh[thresh].append((token1, token2))
        elif score>=0.6:
            for thresh in thresholds[2:]:
                similar_word_based_on_thresh[thresh].append((token1, token2))
        elif score>=0.7:
            for thresh in thresholds[3:]:
                similar_word_based_on_thresh[thresh].append((token1, token2))
        elif score>=0.8:
            similar_word_based_on_thresh[0.8].append((token1, token2))
        # print(score)
    return similar_word_based_on_thresh

thresholds = [0.4, 0.5, 0.6, 0.7, 0.8]

glove_path = '50/glove/hi-d50-glove.txt'
# glove_vec = glove_preprocessing(glove_path)

with open('glove_vec.pickle', 'rb') as handle:
    glove_vec = pickle.load(handle)

similar_word_based_on_thresh = get_glove_similar_words(glove_vec, thresholds, q1, q2)
print(len(glove_vec.keys()))

cbow_path1 = '50/cbow/hi-d50-m2-cbow.model.trainables.syn1neg.npy'
cbow_path2 = '50/cbow/hi-d50-m2-cbow.model.wv.vectors.npy'
word2vec_cbow1 = np.load(cbow_path1)
word2vec_cbow2 = np.load(cbow_path2)
print(word2vec_cbow1)
print(word2vec_cbow2)
print(len(word2vec_cbow2))

# word2vec = KeyedVectors.load_word2vec_format('50/cbow/hi-d50-m2-cbow.model', binary=False, unicode_errors='ignore')
# embedd_dim = word2vec.vector_size
# print(word2vec)
