#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ACTIVATE IR env 


import codecs
import numpy as np
from numpy.linalg import norm
import pickle
import argparse
import time

import gensim
from gensim.models import Word2Vec


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

def get_accuracy(y_pred, y_true, thresholds):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    accuracy=[]
    for t in thresholds:
        y_pred1 = np.where(y_pred>=t, 1, 0)
        y_true1 = np.where(y_true>=t, 1, 0)
        acc = len(np.where(y_pred1==y_true1)[0])/len(y_true)
        accuracy.append(acc)
    return accuracy


def get_glove_similar_words(glove_vec, thresholds, q1, q2, ground_truth):
    similar_word_based_on_thresh = {t:[] for t in thresholds}
    scores_array=[]
    for token1, token2, gd in zip(q1, q2, ground_truth):
        if token1 in glove_vec.keys():
            # print("First Token found: ", token1)
            # print("First Token vec: ", glove_vec[token1])
            # count+=1
            v1 = glove_vec[token1]
        else:
            print("First Token not found: ", token1)
        if token2 in glove_vec.keys():
            # print("Second Token found: ", token2)
            # print("Second Token vec: ", glove_vec[token2])
            # count+=1
            v2 = glove_vec[token2]
        else:
            print("Second Token not found: ", token2)
        score = get_cosine_sim_score(v1, v2)*10
        if score>=thresholds[0] and gd>=thresholds[0]:
            similar_word_based_on_thresh[thresholds[0]].append((token1, token2))
        if score>=thresholds[1] and gd>=thresholds[0]:
            similar_word_based_on_thresh[thresholds[1]].append((token1, token2))
        if score>=thresholds[2] and gd>=thresholds[0]:
            similar_word_based_on_thresh[thresholds[2]].append((token1, token2))
        if score>=thresholds[3] and gd>=thresholds[0]:
            similar_word_based_on_thresh[thresholds[3]].append((token1, token2))
        if score>=thresholds[4] and gd>=thresholds[0]:
            similar_word_based_on_thresh[thresholds[4]].append((token1, token2))
        scores_array.append(score)
    # print("scores_array: ", scores_array)
    # print("ground_truth: ", ground_truth)
    acc = get_accuracy(scores_array, ground_truth, thresholds)
    return similar_word_based_on_thresh, acc

def get_word2vec_fasttext_similar_words(model, thresholds, q1, q2, ground_truth):
    scores_array=[]
    similar_word_based_on_thresh = {t:[] for t in thresholds}
    for token1, token2, gd in zip(q1, q2, ground_truth):
        # print("First Token found: ", token1)
        v1 = model.wv[token1]
        # print("First vec: ", v1)

        # print("Second Token found: ", token2)
        v2 = model.wv[token2]
        # print("Second vec: ", v2)

        score = get_cosine_sim_score(v1, v2)*10
        if score>=thresholds[0] and gd>=thresholds[0]:
            similar_word_based_on_thresh[thresholds[0]].append((token1, token2))
        if score>=thresholds[1] and gd>=thresholds[0]:
            similar_word_based_on_thresh[thresholds[1]].append((token1, token2))
        if score>=thresholds[2] and gd>=thresholds[0]:
            similar_word_based_on_thresh[thresholds[2]].append((token1, token2))
        if score>=thresholds[3] and gd>=thresholds[0]:
            similar_word_based_on_thresh[thresholds[3]].append((token1, token2))
        if score>=thresholds[4] and gd>=thresholds[0]:
            similar_word_based_on_thresh[thresholds[4]].append((token1, token2))
        scores_array.append(score)
    # print("scores_array: ", scores_array)
    # print("ground_truth: ", ground_truth)
    acc = get_accuracy(scores_array, ground_truth, thresholds)
    return similar_word_based_on_thresh, acc

def save_similar_word(Similar_word, name):
    file1 = open('output/'+name+'.txt',"w")
    with file1 as f:
        print(Similar_word, file=f)

if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--word_similarity_data",required=True, help="path to word similarity datasets")
    ap.add_argument("-gof", "--glove_flag", type=int, default=0, help="Do you want to extract glove vec from scratch \
                                                                or you want to use existing glove vec available in the utils folder")
    ap.add_argument("-g", "--glove_path",required=True, help="path to glove pretrained vec")
    ap.add_argument("-gd", "--glove_dict_path",required=True, help="path to glove dict pretrained vec")
    ap.add_argument("-c", "--cbow_path",required=True, help="path to word2vec (cbow) model")
    ap.add_argument("-s", "--sg_path",required=True, help="path to word2vec (sg) model")
    ap.add_argument("-f", "--fasttext_path",required=True, help="path to fasttext model")

    args = vars(ap.parse_args())
    
    start_time = time.time()
    thresholds = [4, 5, 6, 7, 8]

    q1, q2=[], []
    ground_truth=[]
    with codecs.open(args["word_similarity_data"], encoding='utf-8') as f:
        for line in f:
            # print(line)
            tokens = line.strip().split(',')
            # print(tokens)
            if len(tokens[0])>0:
                q1.append(tokens[0])
                q2.append(tokens[1])
                ground_truth.append(float(tokens[2]))
    print("First query vector: ", q1)
    print("Second query vector: ", q2)
    print("Ground Truths: ", ground_truth)

    ####################### GLOVE ###############################
    if args["glove_flag"]:
        glove_path = args["glove_path"] #'50/glove/hi-d50-glove.txt'
        print("Glove flag is working")
        # glove_vec = glove_preprocessing(glove_path)
    else:
        glove_vec_path = args["glove_dict_path"]
        with open(glove_vec_path, 'rb') as handle:
            glove_vec = pickle.load(handle)

    print("============ Using Glove =============")
    glove_similar_word, acc = get_glove_similar_words(glove_vec, thresholds, q1, q2, ground_truth)
    print("Similar Word using glove: ", glove_similar_word)
    for i, a in zip(thresholds, acc):
        print("Accuracy on threshold={}: {}".format(i, round(a, 3)))
    save_similar_word(glove_similar_word, 'glove_similar_word')

    # ####################### Word2Vec (CBOW) ###############################
    print("============ Word2Vec (CBOW) =============")
    cbow_path = args["cbow_path"]
    model = Word2Vec.load(cbow_path)
    cbow_similar_word, acc = get_word2vec_fasttext_similar_words(model, thresholds, q1, q2, ground_truth)
    print("Similar Word using Word2vec(cbow): ", cbow_similar_word)
    for i, a in zip(thresholds, acc):
        print("Accuracy on threshold={}: {}".format(i, round(a, 3)))
    save_similar_word(cbow_similar_word, 'cbow_similar_word')

    # ####################### Word2Vec (SkipGram) ###############################
    print("============ Word2Vec (SG) =============")
    sg_path = args["sg_path"]
    model = Word2Vec.load(sg_path)
    sg_similar_word , acc= get_word2vec_fasttext_similar_words(model, thresholds, q1, q2, ground_truth)
    print("Similar Word using Word2vec(skipgram): ", sg_similar_word)
    for i, a in zip(thresholds, acc):
        print("Accuracy on threshold={}: {}".format(i, round(a, 3)))
    save_similar_word(sg_similar_word, 'sg_similar_word')

    ####################### Fasttext ###############################
    print("============ Fasttext =============")
    fasttext_path = args["fasttext_path"]
    model = Word2Vec.load(fasttext_path)
    fasttext_similar_word, acc = get_word2vec_fasttext_similar_words(model, thresholds, q1, q2, ground_truth)
    print("Similar Word using fasttext: ", fasttext_similar_word)
    for i, a in zip(thresholds, acc):
        print("Accuracy on threshold={}: {}".format(i, round(a, 3)))
    save_similar_word(fasttext_similar_word, 'fasttext_similar_word')

    print("Total Time to run Q1: {} sec".format(round(time.time()-start_time, 3)))