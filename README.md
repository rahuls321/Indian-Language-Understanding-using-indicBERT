# Indian-Language-Understanding-using-indicBERT

Assignment 2 (Language - Hindi)

<!-- Name - Rahul Kumar
Roll No. - 21111069
Prog. - Mtech CSE
Mail - rahulkumar21@iitk.ac.in
Contact - +91-7761937143
 -->

Python version used - 3.8.12

Assignment questions - https://hello.iitk.ac.in/sites/default/files/cs657a22/assignments/problems/749c29d6b8920ba1ac0d5f16e5e48d4934dba76da430c5af15b7a89d380bd8ad_assignment2.pdf

*** NOTE *** 
1. Make sure you are connected with internet. (If you are running on cse server run authenticator.py file to bypass the firewall)
2. Make sure glove_vec.pickle file is downloaded in the utils folder. https://drive.google.com/drive/folders/1jHs1KqWFghTJ9OdXj1fdDP5FiVaQMDxm?usp=sharing
3. Put every data inside the data folder (Download from here -  
    a. Pretrained-word vectors - https://www.cfilt.iitb.ac.in/~diptesh/embeddings/monolingual/non-contextual/
    b. Word similarity datasets - https://drive.google.com/drive/folders/1VovzSE1-zXH0bKCar2M8peL4-62BSlZJ?usp=sharing
    c. NER datasets - https://drive.google.com/file/d/1S5TOqIC37dxWCeQbA9VpplXOGAB7cIMV/view?usp=sharing
    d. Hindi Corpora - https://indicnlp.ai4bharat.org/corpora/
4. Change datapaths accordingly

This folder contains following directories and files 
1. data - contains all the data used in this assignment 2
    a.  Pretrained-word vectors- Used 50d pretrained word vectors.
    b.  Word similarity datasets - contains a set of pairs of two hindi words
    c.  NER datasets - contains list of hindi words and their tags.
    d.  Hindi Corpora - 1.8 B tokens hindi corpora. (Size - approx 22 GB)
2. utils - It contains the necessary file which takes time to build like glove_vec
3. output - a folder contains all the generated outputs for all 3 questions.
    a. For Q1, all the outputs corresponding to different dimensions and different pretrained vector
    b. loss.jpg - Plot Training loss vs Validation acc for Q2
    c. log.txt - Logs of the training for the NER tasks for Q2.
    d. char, syllable, word - 3 folders for Q3 outputs.
4. q1.py - This is file for first question in the assignment which majorily doing the task of word similarity on different thresholds.
5. q2.py - Building NER model for Hindi language.
6. q3.py - This file is for finding all the most frequent char, word and syllable of unigrams, bigrams, trigrams, quadrigrams.
7. run.sh - This is the file that contains all the variable parameters mentioned in the below section.
8. Makefile - There are two commands in the makefile one is "install", "run"
    a.  make install - install all the required packages and download the drive files (please follow the drive link if you're not able to download from the make install)
    b.  make run - will run the whole assignments

*** run.sh is the top-level script that runs the entire assignment. ***

### To run the entire assignment, go to home directory where this README file is there and use following command
##  $ make install 
##  $ make run 

These are the variables that I'm passing as an arguments in the program. [ change accordingly ]

### Q1
1. glove_path="50/glove/hi-d50-glove.txt" #glove pretrained word vector
2. glove_dict_path="utils/glove_vec.pickle" #extracted all the word vector from glove pretrained vec files (available here to download - drive link mentioned above)
3. cbow_path="50/cbow/hi-d50-m2-cbow.model" #Word2vec(cbow) pretrained word vector
4. sg_path="50/sg/hi-d50-m2-sg.model" #Word2vec(sg) pretrained word vector
5. fasttext_path="50/fasttext/hi-d50-m2-fasttext.model" #fasttext pretrained word vector
6. word_similarity_data="data/Word_similarity/hindi.txt" #Word similarity hindi text files
7. glove_flag=0 #flag which is used to verify whether you wanted to extract glove vec from original files or you wanted to use existing extracted glove vec provided in the above link.

### Q2
8. epochs=10 
9. batch_size=32
10. datapath='data/hindi_ner/hi_train.conll' #Data to train the model for NER tasks

### Q3
11. ai_corpus_input_path = 'data/ai_corpus/data/hi/hi.txt' #Hindi corpus 
12. ai_corpus_token_path = 'data/ai_corpus/data/hi/hi.tok.txt' #Hindi corpus in token forms. (as per given on the websites)


## Remarks
### Q1
1. For Q1, I'v considered 50d, 100d vectors (glove, word2vec, fasttext) to get similarity score of two words
2. These thresholds have been considered while finding the accuracy --> thresholds = [4, 5, 6, 7, 8]
3. I have also saved the similar word based on different thresholds as per output format in the 'output' folder. Please check

### Q2
1. NER tasks Code implemented in pytorch.
2. In preprocessing, extract complete sentence by appending all the subwords based on ID given in the data.
3. Remove extra or special characters by checking whether the particular word is hindi or not by one function defined in the code.
4. For NER tasks, there should be a seperate tag corresponding to each word present in the sentence. For that, we need to normalize all labels by padding with 0 and also the hindi word tokens to make every sentence in equal size.
5. Then I used AutoModelForTokenClassification model from transformers to train the model.
6. Reported all the results and also provided the log file to check how is training happening. Provided one plot as well between training loss vs validation accuracy.

### Q3
1. Considered this website (https://jrgraphix.net/r/Unicode/0900-097F) to find all the unicode of the hindi character
2. Considered halant character seperately in unigram, bigram
3. All the unigrams, bigrams, trigrams, and quadrigrams are saved in the output folder for all three char, word, and syllable seperately.
4. For Zipfian Distribution, based on plot whichever follows straight line in the plot of log(frequency) vs log(rank) will follow zipfian distribution
    a. For char: trigram and quadrigram follows zipfian distribution.
    b. For syllable: bigram, trigram and quadrigram follows zipfian distribution
    c. For word: unigram, bigram, trigram and quadrigram all follows zipfian distribution.

### Incase you face any issue in running the code, just let me know here - rahulkumar21@iitk.ac.in
