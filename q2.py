

# ACTIVATE env_tf2.0 env 


from base64 import encode
from cProfile import label
# from imp import cache_from_source
from lib2to3.pgen2 import token
from lib2to3.pgen2.tokenize import tokenize
import sched

from transformers import AutoModel, AutoTokenizer, AdamW, AutoModelForTokenClassification, get_linear_schedule_with_warmup
from keras.preprocessing.sequence import pad_sequences
import numpy as np
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns
import argparse
import time

import torch
import gc
gc.collect()
torch.cuda.empty_cache()
# torch.cuda.memory_summary(device=None, abbreviated=False)
from torch import nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, random_split
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler

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
    ner_tag_label, labels=[], []
    uniq_labels=[]
    max_seq_len = 0
    sent_tokens_dict={}
    with open(datapath, mode='r', encoding='utf-8') as input_file:
        lines = input_file.readlines()
        for line in lines:
            # print(line)
            query = line.strip().split('\t')[0].split(' ')
            # print("query: ", query)
            if '#' in query and 'id' in query: 
                sent_id = query[2]
                sent_tokens=[]
                sent_labels=[]
                sent_tokens_dict[query[2]] = sent_tokens
                continue
            if len(query[0])>0 and is_hindi(query[0]):
                # print(query[0], end=' ')
                # print(query[-1])
                ner_token_dict[query[0]] = query[-1]
                sent_tokens.append(query[0])
                label=query[-1]
                sent_tokens_dict[sent_id].append(query[0])

                # uniq_labels.append(label)
                labels.append(label)
                sent_labels.append(label)
                hi_text = ' '.join(sent_tokens)

            if len(query[0])==0:
                ner_tag_label.append(sent_labels)
                hi_text = ' '.join(sent_tokens)
                # print("hi_text: ", hi_text)
                encoded_dict = tokenizer(hi_text, return_tensors="pt")
                inp_ids = encoded_dict["input_ids"].numpy()[0]
                input_ids.append(inp_ids)
                attention_masks.append(encoded_dict["attention_mask"].numpy()[0])
                max_seq_len = max(max_seq_len, len(inp_ids))

    # print("sent_tokens_dict: ", sent_tokens_dict)
    # print("labels: ", ner_tag_label)
    print("max_seq_len: ", max_seq_len)
    uniq_labels = np.unique(labels)
    uniq_labels = np.insert(uniq_labels, 0, 'PAD') #Padding token
    print("Unique labels: ", uniq_labels)
    labels_dict = {label: i for i, label in enumerate(uniq_labels)}
    # print(labels_dict)
    labels = [[labels_dict[l] for l in label]for label in ner_tag_label]
    # print(labels[:10])
    pad_input_ids = pad_sequences(input_ids, maxlen=max_seq_len, value=0, dtype="long", truncating="post", padding="post")
    pad_attention_masks = pad_sequences(attention_masks, maxlen=max_seq_len, value=0, dtype="long", truncating="post", padding="post")
    pad_labels = pad_sequences(labels, maxlen=max_seq_len, value=0, dtype="long", truncating="post", padding="post")
    # print(pad_labels)
    # print("Labels: ", labels)
    print("input_ids: ", pad_input_ids)
    print("input_ids shape: ", pad_input_ids.shape)
    print("attention_masks: ", pad_attention_masks)
    print("attention_masks shape: ", pad_attention_masks.shape)
    print("Labels: ", labels[:10])
    print("Labels shape: ", len(labels))
    return pad_input_ids, pad_attention_masks, pad_labels, labels_dict

class NER_BERT(nn.Module):
    def __init__(self, num_classes):
        super(NER_BERT, self).__init__()
        self.baseline = AutoModelForTokenClassification.from_pretrained('ai4bharat/indic-bert', cache_dir="/data/rahulk/transformer-models", num_labels=num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_masks: torch.Tensor, 
        labels=torch.Tensor
    ) -> torch.Tensor:

        
        outputs = self.baseline.forward(input_ids=input_ids, attention_mask=attention_masks, labels=labels)
        out = F.softmax(outputs.logits, dim=2)
        loss = outputs[0]
        return loss, out


def flat_accuracy(preds, labels):
    flat_preds = np.array(preds)
    flat_labels = np.array(labels)
    return np.sum(flat_preds == flat_labels)/len(flat_labels)

def train(model,train_dataloader, scheduler):
    model.train()
    total_loss = 0
    for step, batch in enumerate(train_dataloader):
       
        # if step % 40 == 0 and not step == 0:
            
        #     # Report progress.
        #     print('  Batch {:>5,}  of  {:>5,}.'.format(step, len(train_dataloader)))
        #     print("Loss: ", loss)

        b_input_ids = batch[0].to(device)
        b_input_mask = batch[1].to(device)
        b_labels = batch[2].to(device)

        model.zero_grad()        

        loss, outputs = model(b_input_ids, b_input_mask, b_labels)
        #print("Loss: ", loss.item())
        total_loss += loss.item()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

    avg_train_loss = total_loss / len(train_dataloader)            
    # loss_values.append(avg_train_loss)

    print("  Average training loss: {0:.2f}".format(avg_train_loss))

    return avg_train_loss

def test(model, testing_loader, tag_values):
    model.eval()
    eval_loss = 0; eval_accuracy = 0
    predictions , true_labels = [], []
    nb_eval_steps, nb_eval_examples = 0, 0
    with torch.no_grad():
        # Predict 
        for batch in testing_loader:

            # Add batch to GPU
            batch = tuple(t.to(device) for t in batch)

            # Unpack the inputs from our dataloader
            ids, mask, targets = batch

            loss, logits = model(ids, mask, labels=targets)
            logits = logits.detach().cpu().numpy()
            label_ids = targets.to('cpu').numpy()
            logits = [list(p) for p in np.argmax(logits, axis=2)]
            label_ids = [p[np.where(p>0)] for p in label_ids]
            logits = [lo[:len(la)] for lo, la in zip(logits, label_ids)]
            logits = [p for pred in logits for p in pred]
            label_ids = [l for label in label_ids for l in label]
            predictions.append(logits)
            true_labels.append(label_ids)
            accuracy = flat_accuracy(logits, label_ids)
            eval_loss += loss.mean().item()
            eval_accuracy += accuracy
            nb_eval_examples += ids.size(0)
            nb_eval_steps += 1
        eval_loss = eval_loss/nb_eval_steps
        pred_tags = [tag_values[p_i] for p in predictions for p_i in p]
        valid_tags = [tag_values[l_i] for l in true_labels for l_i in l]
        f1score = f1_score(pred_tags, valid_tags, average='micro')
        print("F1-Score: {}".format(f1score))
    return f1score

def plot_loss(train_loss, val_accuracy):

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12,6)

    # Plot the learning curve.
    plt.plot(train_loss, 'b-o', label='Training loss')
    plt.plot(val_accuracy, 'r-o', label='Validation Accuracy')

    # Label the plot.
    plt.title("Training loss & Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.savefig("output/loss.jpg")
    plt.show()


if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("--datapath", required=True, help="Hindi corpora data for NER tasks")
    ap.add_argument("--batch_size", type=int, default=64, help="Batch size used to train model")
    ap.add_argument("--epochs", type=int, default=10, help="Epochs used to train model")

    args = vars(ap.parse_args())
    start_time = time.time()

    batch_size = args["batch_size"]
    datapath=args["datapath"]
    tokenizer = AutoTokenizer.from_pretrained('ai4bharat/indic-bert', cache_dir="/data/rahulk/transformer-models")

    print("Preprocessing Started ------------")
    input_ids, attention_masks, labels, labels_dict = preprocessing(datapath)
    # exit()
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

    model = NER_BERT(num_classes)
    model = model.to(device)

    # Load the AdamW optimizer
    optimizer = AdamW(model.parameters(), lr = 5e-5, eps = 1e-8)
    tag_values = list(labels_dict.keys())
    epochs=args["epochs"]
    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer, 
                                                num_warmup_steps = 0,
                                                num_training_steps = total_steps)
    train_loss, val_accuracy=[], []
    print('Training...')
    for epoch_i in range(0, epochs):
        print('======== Epoch {:} / {:} ========'.format(epoch_i + 1, epochs))
        avg_loss = train(model, train_dataloader, scheduler)
        train_loss.append(avg_loss)
        # print("Training: ")
        # test(model, train_dataloader, tag_values)
        print("Validation: ")
        f1score=test(model, val_dataloader, tag_values)
        val_accuracy.append(f1score)

    print("Training: ")
    score=test(model, train_dataloader, tag_values)
    print("Validation: ")
    score=test(model, val_dataloader, tag_values)
    print("Testing: ")
    score=test(model, test_dataloader, tag_values)

    plot_loss(train_loss, val_accuracy)

    print("Total Time to run Q2: {} sec".format(round(time.time()-start_time, 3)))