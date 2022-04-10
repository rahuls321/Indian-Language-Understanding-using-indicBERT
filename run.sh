# export PATH=/usr/local/cuda-11.2/bin/:$PATH
# export LD_LIBRARY_PATH=/data/rahulk/cuda/lib64:${LD_LIBRARY_PATH}
# export LD_LIBRARY_PATH=/usr/local/cuda-11.2/lib64/:$LD_LIBRARY_PATH

##Q1
glove_path="50/glove/hi-d50-glove.txt"
glove_dict_path="utils/glove_vec.pickle"
cbow_path="50/cbow/hi-d50-m2-cbow.model"
sg_path="50/sg/hi-d50-m2-sg.model"
fasttext_path="50/fasttext/hi-d50-m2-fasttext.model"
word_similarity_data="data/Word_similarity/hindi.txt"
glove_flag=0

#Q2
epochs=10
batch_size=32
datapath='data/hindi_ner/hi_train.conll'

#Q3
ai_corpus_input_path = 'data/ai_corpus/data/hi/hi.txt'
ai_corpus_token_path = 'data/ai_corpus/data/hi/hi.tok.txt'

# python q1.py -d $word_similarity_data -g $glove_path -gd $glove_dict_path -gof $glove_flag -c $cbow_path -s $sg_path -f $fasttext_path
python q2.py --batch_size $batch_size --epochs $epochs --datapath $datapath
# python q3.py --ai_corpus_input_path $ai_corpus_input_path --ai_corpus_token_path $ai_corpus_token_path