from indicnlp.tokenize.indic_tokenize import trivial_tokenize
from indicnlp.normalize.indic_normalize import IndicNormalizerFactory
from enum import unique
from indicnlp.langinfo import *
import regex
from nltk import word_tokenize, ngrams
import matplotlib.pyplot as plt
import argparse
import numpy as np

import time

def process_sent(sent):
    normalized = normalizer.normalize(sent)
    processed = ' '.join(trivial_tokenize(normalized, lang))
    return processed

def is_hindi(character):
    maxchar = max(character)
    if u'\u0900' <= maxchar <= u'\u097f':
        return True
    else:
      return False

def preprocessing(input_path, output_path):
    print("Preprocessing of the Hindi corpus started: ")
    st = time.time()
    with open(input_path, 'r', encoding='utf-8') as in_fp,\
        open(output_path, 'w', encoding='utf-8') as out_fp:
        for line in in_fp.readlines():
            print(line)
            sent = line.rstrip('\n')
            toksent = process_sent(sent)
            out_fp.write(toksent)
            out_fp.write('\n')
    print(f"Total time takes to preprocess: {time.time() - st}")
#Removing extra characters like special char, numbers, comma, full stop(hindi)
def hindi_preprocessing(hindi_text):
    h_tokens = hindi_text.split()
    up_tokens=[]
    for t in h_tokens:
        if is_hindi(t) and t!=u'।':
            up_tokens.append(t)
    return ' '.join(up_tokens)

def get_char_ngrams(line, suffixes, halant, unigrams_freq, bigrams_freq, trigrams_freq, quadrigrams_freq):
    all_char=[]
    unknown=[]
    lang='hi'

    # print("Unigram preprocessing started: ")
    prev_char=''
    for char in line:
        c = "\n".join(regex.findall(r'\X', char, regex.U))
        if c==' ' or len(c)>1: 
            if len(c)>1: print("COntinue this c: ", c)
            continue
        if c in suffixes.keys():
            c = suffixes[c]
        all_char.append(c)
    # print(all_char)
    for j,c in enumerate(all_char):
        # if len(c)>1 or len(c)==0: print(c)
        if c!=u"अं" and c!=u"अ:" and not is_vowel(c,lang) and not is_consonant(c, lang):
            unknown.append(c)
            continue
        # print(c, end=' ')
        # print(len(c))
        if j+1<len(all_char) and len(all_char[j+1])>0 and len(all_char[j+1])<2 and c!=u"अं" and c!=u"अ:" and is_consonant(all_char[j+1], lang) and is_consonant(c, lang):
                c += halant
                aa = u"अ"
                if aa in unigrams_freq.keys(): unigrams_freq[aa]+=1
                else: unigrams_freq[aa]=1
        if c in unigrams_freq.keys():
            unigrams_freq[c]+=1
        else:
            unigrams_freq[c]=1

    all_char=[]
    unknown=[]

    for char in line:
        c = "\n".join(regex.findall(r'\X', char, regex.U))
        if c==' ' or len(c)>1: 
            if len(c)>1: print("COntinue this c: ", c)
            continue
        all_char.append(c)

    # print("Bigram preprocessing started: ")
    # print(all_char)
    for j in range(len(all_char)):
        if all_char[j]!=u"अं" and all_char[j]!=u"अ:" and not is_vowel(all_char[j],lang) and not is_consonant(all_char[j], lang):
            unknown.append(all_char[j])
            continue
        #This is used to check if next character is vowel or not based on that only join the two words to make bigrams 
        #if next character is not vowel next consonant is itself a biagram as halant+vowel
        if j+1<len(all_char) and all_char[j]!=u"अं" and all_char[j]!=u"अ:" and all_char[j+1]!=u"अं" and all_char[j+1]!=u"अ:" and is_vowel(all_char[j+1], lang):
            c = ''.join(all_char[j:j+2])
        else:
            c = all_char[j]
        # print(c)
        if c in bigrams_freq.keys():
            bigrams_freq[c]+=1
        else:
            bigrams_freq[c]=1


    # print("Trigram preprocessing started: ")
    for j in range(len(all_char)):
        if all_char[j]!=u"अं" and all_char[j]!=u"अ:" and not is_vowel(all_char[j],lang) and not is_consonant(all_char[j], lang):
            unknown.append(all_char[j])
            continue
        if(j+3<len(all_char)): c = ''.join(all_char[j:j+3])
        else: continue
        # print(c)
        if c in trigrams_freq.keys():
            trigrams_freq[c]+=1
        else:
            trigrams_freq[c]=1

    # print("Quadrigram preprocessing started: ")
    for j in range(len(all_char)):
        if all_char[j]!=u"अं" and all_char[j]!=u"अ:" and not is_vowel(all_char[j],lang) and not is_consonant(all_char[j], lang):
            unknown.append(all_char[j])
            continue
        if(j+4<len(all_char)): c = ''.join(all_char[j:j+4])
        else: continue
        # print(c)
        if c in quadrigrams_freq.keys():
            quadrigrams_freq[c]+=1
        else:
            quadrigrams_freq[c]=1
    
    return unigrams_freq, bigrams_freq, trigrams_freq, quadrigrams_freq


def get_word_ngrams(line, unigrams_freq, bigrams_freq, trigrams_freq, quadrigrams_freq):
    # Word ngrams.
    i=0

    ########## Unigram #############
    unigram = list(ngrams(word_tokenize(line), 1))
    st = [' '.join(t) for t in unigram]
    # print("unigram: ", unigram)
    # print("unigram: ", st)

    for uni in st:
        if uni in unigrams_freq.keys():
            unigrams_freq[uni]+=1
        else:
            unigrams_freq[uni]=1

    ########## Bigram #############
    bigram = list(ngrams(word_tokenize(line), 2))
    st = [' '.join(t) for t in bigram]
    # print("biagram: ", st)

    for bi in st:
        if bi in bigrams_freq.keys():
            bigrams_freq[bi]+=1
        else:
            bigrams_freq[bi]=1

    ########## Trigram #############
    trigram = list(ngrams(word_tokenize(line), 3))
    st = [' '.join(t) for t in trigram]
    # print("trigram: ", st)

    for tri in st:
        if tri in trigrams_freq.keys():
            trigrams_freq[tri]+=1
        else:
            trigrams_freq[tri]=1

    ########## Quadrigram #############
    quadrigram = list(ngrams(word_tokenize(line), 4))
    st = [' '.join(t) for t in quadrigram]
    # print("quadrigram: ", st)

    for quad in st:
        if quad in quadrigrams_freq.keys():
            quadrigrams_freq[quad]+=1
        else:
            quadrigrams_freq[quad]=1
        
    return unigrams_freq, bigrams_freq, trigrams_freq, quadrigrams_freq

def get_syllb(line):
    # https://jrgraphix.net/r/Unicode/0900-097F
    signs = [u'\u0902', u'\u0903', u'\u093e', u'\u093f', u'\u0940',
            u'\u0941', u'\u0942', u'\u0943', u'\u0944', u'\u0946',
            u'\u0947', u'\u0948', u'\u094a', u'\u094b', u'\u094c',
            u'\u094d']
    limiters = ['.', '\"', '\'', '`', '!', ';', ', ', '?']
    halant = u'\u094d'
    lst_chars = []

    for char in line:
        if char==' ': continue
        if char in limiters:
            lst_chars.append(char)
        elif char in signs:
            try:
                lst_chars[-1] = lst_chars[-1] + char
            except IndexError:
                lst_chars.append(char)
        else:
            try:
                if lst_chars[-1][-1] == halant:
                    lst_chars[-1] = lst_chars[-1] + char
                else:
                    lst_chars.append(char)
            except IndexError:
                lst_chars.append(char)
    # print(lst_chars)
    return lst_chars

def get_syllb_ngrams(line, unigrams_freq, bigrams_freq, trigrams_freq, quadrigrams_freq):

    ########## Unigram #############
    st = get_syllb(line)
    for uni in st:
        if uni in unigrams_freq.keys():
            unigrams_freq[uni]+=1
        else:
            unigrams_freq[uni]=1

    ########## Bigram #############
    st = get_syllb(line)
    st = [''.join(st[i:i+2]) for i,t in enumerate(st)]
    # print("bi: ", st)
    for bi in st:
        if bi in bigrams_freq.keys():
            bigrams_freq[bi]+=1
        else:
            bigrams_freq[bi]=1

    ########## Trigram #############
    st = get_syllb(line)
    st = [''.join(st[i:i+3]) for i,t in enumerate(st)]
    # print("tri: ", st)
    for tri in st:
        if tri in trigrams_freq.keys():
            trigrams_freq[tri]+=1
        else:
            trigrams_freq[tri]=1

    ########## Quadrigram #############
    st = get_syllb(line)
    st = [''.join(st[i:i+4]) for i,t in enumerate(st)]
    # print("quad: ", st)
    for quad in st:
        if quad in quadrigrams_freq.keys():
            quadrigrams_freq[quad]+=1
        else:
            quadrigrams_freq[quad]=1
        
    return unigrams_freq, bigrams_freq, trigrams_freq, quadrigrams_freq

def get_zipfian_dist_plot(ngrams, folder, choice):
    # plt.bar(*zip(*ngrams.items()))
    # plt.xticks([])
    # plt.savefig('output/'+folder+'/'+choice+'.jpg')
    # plt.close()

    file1 = open('output/'+folder+'/'+choice+'.txt',"w")
    with file1 as f:
        print(ngrams, file=f)

    y = np.log(list(ngrams.values()))
    x1 = np.log(np.arange(len(ngrams.items()))+1)
    plt.plot(x1, y)
    plt.xlabel("log(rank)")  # add X-axis label
    plt.ylabel("log(freq)")  # add Y-axis label
    plt.title("Zipfian Distribution - log(freq) vs log(rank)")  # add title
    plt.savefig('output/'+folder+'/'+choice+'.jpg')
    plt.close()


def get_ngrams(input_path, suffixes, halant):
    i=0
    c_unigrams_freq, c_bigrams_freq, c_trigrams_freq, c_quadrigrams_freq={},{},{},{}
    w_unigrams_freq, w_bigrams_freq, w_trigrams_freq, w_quadrigrams_freq={},{},{},{}
    s_unigrams_freq, s_bigrams_freq, s_trigrams_freq, s_quadrigrams_freq={},{},{},{}
    with open(input_path, 'r', encoding='utf-8') as in_fp:
        for line in in_fp.readlines():
            line = hindi_preprocessing(line)
            print(i, end=' ')
            print(line)
            c_unigrams_freq, c_bigrams_freq, c_trigrams_freq, c_quadrigrams_freq = get_char_ngrams(line, suffixes, halant, c_unigrams_freq, c_bigrams_freq, c_trigrams_freq, c_quadrigrams_freq)
            w_unigrams_freq, w_bigrams_freq, w_trigrams_freq, w_quadrigrams_freq = get_word_ngrams(line, w_unigrams_freq, w_bigrams_freq, w_trigrams_freq, w_quadrigrams_freq)
            s_unigrams_freq, s_bigrams_freq, s_trigrams_freq, s_quadrigrams_freq = get_syllb_ngrams(line, s_unigrams_freq, s_bigrams_freq, s_trigrams_freq, s_quadrigrams_freq)
            if i==19425000:  #19425000: 
                # print(unigrams_freq)
                break
            i+=1
    c_unigrams_freq=sorted(c_unigrams_freq.items(), key=lambda x: x[1], reverse=True)[:100]
    c_bigrams_freq=sorted(c_bigrams_freq.items(), key=lambda x: x[1], reverse=True)[:100]
    c_trigrams_freq=sorted(c_trigrams_freq.items(), key=lambda x: x[1], reverse=True)[:100]
    c_quadrigrams_freq=sorted(c_quadrigrams_freq.items(), key=lambda x: x[1], reverse=True)[:100]

    w_unigrams_freq=sorted(w_unigrams_freq.items(), key=lambda x: x[1], reverse=True)[:100]
    w_bigrams_freq=sorted(w_bigrams_freq.items(), key=lambda x: x[1], reverse=True)[:100]
    w_trigrams_freq=sorted(w_trigrams_freq.items(), key=lambda x: x[1], reverse=True)[:100]
    w_quadrigrams_freq=sorted(w_quadrigrams_freq.items(), key=lambda x: x[1], reverse=True)[:100]

    s_unigrams_freq=sorted(s_unigrams_freq.items(), key=lambda x: x[1], reverse=True)[:100]
    s_bigrams_freq=sorted(s_bigrams_freq.items(), key=lambda x: x[1], reverse=True)[:100]
    s_trigrams_freq=sorted(s_trigrams_freq.items(), key=lambda x: x[1], reverse=True)[:100]
    s_quadrigrams_freq=sorted(s_quadrigrams_freq.items(), key=lambda x: x[1], reverse=True)[:100]
    return (c_unigrams_freq, c_bigrams_freq, c_trigrams_freq, c_quadrigrams_freq), (w_unigrams_freq, w_bigrams_freq, w_trigrams_freq, w_quadrigrams_freq), (s_unigrams_freq, s_bigrams_freq, s_trigrams_freq, s_quadrigrams_freq)

if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--ai_corpus_input_path", help="ai_corpus_input_path")
    ap.add_argument("--ai_corpus_token_path", help="ai_corpus_token_path")
    ap.add_argument("--preprocess_corpora", type=int, default=0, help="ai_corpus_token_path")
    args = vars(ap.parse_args())
    start_time = time.time()
    lang = 'hi'
    normalizer_factory = IndicNormalizerFactory()
    normalizer = normalizer_factory.get_normalizer(lang)
    if args["preprocess_corpora"]:
        preprocessing(args["ai_corpus_input_path"], args["ai_corpus_token_path"])

    exit()
    output_path=args["ai_corpus_token_path"]
    suffixes = {u'\u093E':u'\u0906', u'\u093F':u'\u0907', u'\u0940':u'\u0908', u'\u0941':u'\u0909', u'\u0942':u'\u090A', u'\u0947':u'\u090F', u'\u0948':u'\u0910', u'\u094B':u'\u0913', u'\u094C':u'\u0914',u"ं":u"अं",u"ः":u"अ:"} 
    #{u"ा":u"अ",u"ि":u"इ",u"ी":u"ई",u"ु":u"उ",u"ू":u"ऊ",u"े":u"ए",u"ै":u"ऐ",u"ो":u"ओ",u"ौ":u"औ",u"ं":u"अं",u"ः":u"अ:"}
    halant = u'\u094D' #u"्"
    # vowels = [u"अ",u"आ",u"इ",u"ई",u"उ",u"ऊ",u"ऋ",u"ए",u"ऐ",u"ओ",u"औ",u"अं",u"अ:"]
    vowels = [u'\u0905', u'\u0906', u'\u0907', u'\u0908', u'\u0909', u'\u090A', u'\u090B', u'\u090F', u'\u0910', u'\u0913', u'\u0914', u"अं",u"अ:"]
    (c_unigrams, c_bigrams, c_trigrams, c_quadrigrams ), (w_unigrams, w_bigrams, w_trigrams, w_quadrigrams), (s_unigrams, s_bigrams, s_trigrams, s_quadrigrams)= get_ngrams(output_path, suffixes, halant)
    print("Top 100 most unigrams char: ", c_unigrams)
    print("Top 100 most bigrams char: ", c_bigrams)
    print("Top 100 most trigrams char: ", c_trigrams)
    print("Top 100 most quadrigrams char: ", c_quadrigrams)

    # get_zipfian_dist_plot(dict(c_unigrams), 'char', 'unigram')
    # get_zipfian_dist_plot(dict(c_bigrams), 'char', 'bigram')
    # get_zipfian_dist_plot(dict(c_trigrams), 'char', 'trigram')
    # get_zipfian_dist_plot(dict(c_quadrigrams), 'char', 'quadrigram')

    print("Top 100 most unigrams word: ", w_unigrams)
    print("Top 100 most bigrams word: ", w_bigrams)
    print("Top 100 most trigrams word: ", w_trigrams)
    print("Top 100 most quadrigrams word: ", w_quadrigrams)

    # get_zipfian_dist_plot(dict(w_unigrams), 'word', 'unigram')
    # get_zipfian_dist_plot(dict(w_bigrams), 'word', 'bigram')
    # get_zipfian_dist_plot(dict(w_trigrams), 'word', 'trigram')
    # get_zipfian_dist_plot(dict(w_quadrigrams), 'word', 'quadrigram')

    print("Top 100 most unigrams syllable: ", s_unigrams)
    print("Top 100 most bigrams syllable: ", s_bigrams)
    print("Top 100 most trigrams syllable: ", s_trigrams)
    print("Top 100 most quadrigrams syllable: ", s_quadrigrams)

    # get_zipfian_dist_plot(dict(s_unigrams), 'syllable', 'unigram')
    # get_zipfian_dist_plot(dict(s_bigrams), 'syllable', 'bigram')
    # get_zipfian_dist_plot(dict(s_trigrams), 'syllable', 'trigram')
    # get_zipfian_dist_plot(dict(s_quadrigrams), 'syllable', 'quadrigram')

    print("Total Time to run Q3: {} sec".format(round(time.time()-start_time, 3)))
