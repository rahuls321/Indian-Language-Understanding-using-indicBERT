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
import pandas as pd

import gensim
from gensim.models import Word2Vec


def get_cosine_sim_score(v1, v2):
    cos_sim = np.dot(v1, v2)/(norm(v1)*norm(v2))
    return cos_sim

def glove_preprocessing(glove_path):
    glove_vec={}
    st = time.time()
    with codecs.open(glove_path, encoding='utf-8', errors='ignore') as f:
        for line in f:
            tokens = line.split()
            token = tokens[0]
            vec = tokens[1:]
            glove_vec[token] = [float(v) for v in vec]
            print(token, end=' ')
            print(glove_vec[token])
    print(f"Total time in glove vec takes in preprocessing: {time.time() - st} sec")
    # with open('glove_vec.pickle', 'wb') as handle:
    #     pickle.dump(glove_vec, handle)
    return glove_vec

def csv_save_file(q1, q2, acc, scores_array, ground_truth, dim, choice, labels, threshold):
    df1 = pd.DataFrame({'Word1':q1, 'Word2':q2, 'Similarity Score':np.round_(scores_array, decimals=1), 'Ground Truth similarity score':ground_truth, 'Label': labels})
    df2 = pd.DataFrame({'Word1':f'Total Accuracy: {round(acc, 3)}', 'Word2':'', 'Similarity Score':'', 'Ground Truth similarity score':'', 'Label':''}, index=df1.columns)
    df = pd.concat([df1, df2])
    df.to_csv(f'output/Q1_{dim}_{choice}_similarity_{threshold}.csv', index=False)


def get_accuracy(q1, q2, y_pred, y_true, dim, choice, thresholds):
    y_pred = np.array(y_pred)
    y_true = np.array(y_true)
    accuracy=[]
    for t in thresholds:
        y_pred1 = np.where(y_pred>=t, 1, 0)
        y_true1 = np.where(y_true>=t, 1, 0)
        acc = len(np.where(y_pred1==y_true1)[0])/len(y_true)
        csv_save_file(q1, q2,acc, y_pred, y_true, dim, choice, y_pred1, t)
        accuracy.append(acc)
    return accuracy


def get_glove_similar_words(glove_vec, thresholds, q1, q2, dim, choice, ground_truth):
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
    acc = get_accuracy(q1, q2, scores_array, ground_truth, dim, choice, thresholds)
    return similar_word_based_on_thresh, acc

def get_word2vec_fasttext_similar_words(model, thresholds, q1, q2, dim, choice, ground_truth):
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
    acc = get_accuracy(q1, q2, scores_array, ground_truth, dim, choice, thresholds)
    return similar_word_based_on_thresh, acc

def save_similar_word(Similar_word, q1, q2, name):
    file1 = open('output/'+name+'.txt',"w")
    with file1 as f:
        print(Similar_word, file=f)

if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--word_similarity_data",required=True, help="path to word similarity datasets")
    ap.add_argument("--dim",required=True, default='50', help="Different Dimensions to use")
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
        glove_path = args["glove_path"]
        print("Glove flag is working")
        glove_vec = glove_preprocessing(glove_path)
    else:
        glove_vec_path = args["glove_dict_path"]
        with open(glove_vec_path, 'rb') as handle:
            glove_vec = pickle.load(handle)

    dimensions = args["dim"].split(',')
    dimensions = [int(x) for x in dimensions]


    for dim in dimensions:
        print(f"======================================= For Dim: {dim} ==============================================")
        for vec_path, vec_type in zip(['', args["cbow_path"], args["sg_path"], args["fasttext_path"]], ['Glove', 'Word2Vec_CBOW', 'Word2Vec_SG', 'Word2Vec_FastText']):
            print(f"===================== {vec_type}, dim:{dim} ===================")
            if vec_type=='Glove':
                similar_word, acc = get_glove_similar_words(glove_vec, thresholds, q1, q2, dim, vec_type, ground_truth)
            else:
                model = Word2Vec.load(vec_path)
                similar_word, acc = get_word2vec_fasttext_similar_words(model, thresholds, q1, q2, dim, vec_type, ground_truth)
            # print(f"Similar Word using {vec_type}: {similar_word}")
            for i, a in zip(thresholds, acc):
                print("Accuracy on threshold={}: {}".format(i, round(a, 3)))

    print("Total Time to run Q1: {} sec".format(round(time.time()-start_time, 3)))