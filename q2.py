

from base64 import encode
from cProfile import label
from imp import cache_from_source
from lib2to3.pgen2 import token
from lib2to3.pgen2.tokenize import tokenize

from transformers import AutoModel, AutoTokenizer, AdamW, AutoModelForTokenClassification
from keras.preprocessing.sequence import pad_sequences
import numpy as np

import torch
from torch import nn
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

batch_size = 32
datapath='/data/rahulk/IR/assign2/NER_datasets/Hindi/wikiann-hi.bio'
tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert', cache_dir="/data/rahulk/transformer-models")
# bert_model = AutoModel.from_pretrained('ai4bharat/indic-bert', cache_dir="/data/rahulk/transformer-models", num_labels=7)

#Checking GPU availability
if torch.cuda.is_available():       
    device = torch.device("cuda")
    print( torch.cuda.device_count())
    print('Available:', torch.cuda.get_device_name(0))
else:
    print('No GPU available, using the CPU instead.')
    device = torch.device("cpu")


def is_hindi(character):
    maxchar = max(character)
    if u'\u0900' <= maxchar <= u'\u097f':
        return True
    else:
      return False

def preprocessing(datapath):
    ner_token_dict={}
    input_ids = []
    attention_masks = []
    labels=[]
    uniq_labels=[]
    max_seq_len = 0
    with open(datapath, mode='r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
        for line in lines:
            query = line.strip().split('\t')[0].split(' ')
            # print(query)
            tokens=[]
            label=''
            flag=0
            for q in query:
                if len(q)>0 and is_hindi(q):
                    # print(q)
                    ner_token_dict[q] = query[-1]
                    tokens.append(q)
                    label=query[-1]
                    flag=1
            if(flag): 
                uniq_labels.append(label)
                labels.append([label] * len(tokens))
            if len(tokens)>0:
                hi_text = ' '.join(tokens)
                print(hi_text)
                encoded_dict = tokenizer(hi_text, return_tensors="pt")
                inp_ids = encoded_dict["input_ids"].numpy()[0]
                input_ids.append(inp_ids)
                attention_masks.append(encoded_dict["attention_mask"].numpy()[0])
                max_seq_len = max(max_seq_len, len(inp_ids))

    print("max_seq_len: ", max_seq_len)
    uniq_labels = np.unique(uniq_labels)
    print("Unique labels: ", uniq_labels)
    labels_dict = {label: i+1 for i, label in enumerate(uniq_labels)}
    # print(labels_dict)
    labels = [[labels_dict[l] for l in label]for label in labels]
    # print(labels[:10])
    pad_input_ids = pad_sequences(input_ids, maxlen=max_seq_len, value=0, dtype="long", truncating="post", padding="post")
    pad_attention_masks = pad_sequences(attention_masks, maxlen=max_seq_len, value=0, dtype="long", truncating="post", padding="post")
    pad_labels = pad_sequences(labels, maxlen=max_seq_len, value=0, dtype="long", truncating="post", padding="post")
    # print(pad_labels)
    # print("Labels: ", labels)
    # print("input_ids: ", pad_input_ids)
    # print("input_ids shape: ", pad_input_ids.shape)
    # print("attention_masks: ", pad_attention_masks)
    # print("attention_masks shape: ", pad_attention_masks.shape)
    # print("Labels: ", labels[:10])
    # print("Labels shape: ", len(labels))
    return pad_input_ids, pad_attention_masks, pad_labels, labels_dict

class NER_BERT(nn.Module):
    def __init__(self, backbone: nn.Module):
        super(NER_BERT, self).__init__()
        self.backbone = backbone

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor,
        num_classes
    ) -> torch.Tensor:

        
        outputs = self.backbone.forward(input_ids=input_ids, attention_mask=attention_masks)
        # final_out = nn.Linear(num_classes)(outputs.last_hidden_state)
        print(outputs)
        print(outputs.logits.shape)
        return outputs

    
input_ids, attention_masks, labels, labels_dict = preprocessing(datapath)

num_classes = len(labels_dict)

input_ids = torch.tensor(input_ids)
attention_masks = torch.tensor(attention_masks)
labels = torch.tensor(labels)

print("input_ids: ", input_ids)
print("input_ids shape: ", input_ids.shape)
print("attention_masks: ", attention_masks)
print("attention_masks shape: ", attention_masks.shape)
print("Labels: ", labels)
print("Labels shape: ", labels.shape)


# Combine the training inputs into a TensorDataset.
dataset = TensorDataset(input_ids, attention_masks, labels)

# Create a 90-10 train-validation split.
t_size = int(0.9 * len(dataset))
test_size = len(dataset) - t_size
train_size = int(0.9 * t_size)
val_size = t_size - train_size

print("Train size: ", train_size)
print("Val size: ", val_size)
print("Test size: ", test_size)


# Divide the dataset by randomly selecting samples.
train_dataset, val_dataset, test_dataset = random_split(dataset, [train_size, val_size, test_size])


train_dataloader = DataLoader(train_dataset, sampler = RandomSampler(train_dataset), batch_size = batch_size)
val_dataloader = DataLoader(val_dataset, sampler = SequentialSampler(val_dataset), batch_size = batch_size)
test_dataloader = DataLoader(test_dataset, sampler = SequentialSampler(test_dataset), batch_size = batch_size)


bert_model  = AutoModelForTokenClassification.from_pretrained('ai4bharat/indic-bert', cache_dir="/data/rahulk/transformer-models", num_labels=num_classes)
model = NER_BERT(bert_model)
scores = model(input_ids[:10], attention_masks[:10], num_classes)
# print(scores)

# Load the AdamW optimizer
optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)