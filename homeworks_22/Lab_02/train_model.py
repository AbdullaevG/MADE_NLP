import torch
import torch.nn as nn
import torch.optim as optim

import torchtext
from torchtext.datasets import TranslationDataset, Multi30k
from torchtext.data import Field, BucketIterator

import spacy

import random
import math
import time

import matplotlib
matplotlib.rcParams.update({'figure.figsize': (16, 12), 'font.size': 14})
import matplotlib.pyplot as plt
from IPython.display import clear_output

from nltk.tokenize import WordPunctTokenizer
from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE


def get_loss_on_train(model, iterator, optimizer, criterion, clip, 
                      train_history=None, valid_history=None, transformer = False):
    model.train() 
    epoch_loss = 0
    history = []
    for i, batch in enumerate(iterator):
        
        src = batch.src
        trg = batch.trg
        
        optimizer.zero_grad()
        
        if transformer:
            
            output = model(src, trg[:-1, :])
            output = output.contiguous().view(-1, output.shape[-1])
            
            trg = trg[1:, :].contiguous().view(-1)
        
        else:
       	    output = model(src, trg)
            output = output[1:].view(-1, output.shape[-1])

            trg = trg[1:].view(-1)
        
        loss = criterion(output, trg)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
        optimizer.step()
        epoch_loss += loss.item()
        
        history.append(loss.cpu().data.numpy())
        
        if (i+1)%10==0:
            fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 8))

            clear_output(True)
            ax[0].plot(history, label='train loss')
            ax[0].set_xlabel('Batch')
            ax[0].set_title('Train loss')
            if train_history is not None:
                ax[1].plot(train_history, label='general train history')
                ax[1].set_xlabel('Epoch')
            if valid_history is not None:
                ax[1].plot(valid_history, label='general valid history')
            plt.legend()
            
            plt.show()
            
    return epoch_loss / len(iterator)


def get_loss_on_val(model, iterator, criterion, transformer = False):
    
    model.eval()
    epoch_loss = 0  
    history = []
    
    with torch.no_grad():
    
        for i, batch in enumerate(iterator):

            src = batch.src
            trg = batch.trg
            
            if transformer:
                output = model(src, trg[:-1, :])
                output = output.view(-1, output.shape[-1])
                trg = trg[1:, :].view(-1)

            else:
                output = model(src, trg, 0)
                output = output[1:].view(-1, output.shape[-1])
                trg = trg[1:].view(-1)

            loss = criterion(output, trg)
            epoch_loss += loss.item()
        
    return epoch_loss / len(iterator)


def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs


N_EPOCHS = 10
CLIP = 1

def train(model,
          model_name,
          train_iterator, 
          valid_iterator, 
          optimizer,
          scheduler,
          criterion,
          CLIP = CLIP,
          n_epochs = N_EPOCHS,
          transformer = False):
    
    best_valid_loss = float('inf')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    train_history = []
    valid_history = []

    for epoch in range(n_epochs):
        start_time = time.time()
        train_loss = get_loss_on_train(model, 
                                       train_iterator, 
                                       optimizer, criterion, 
                                       CLIP, train_history, 
                                       valid_history, 
                                       transformer = transformer)
        
        valid_loss = get_loss_on_val(model, valid_iterator, criterion, 
                                     transformer = transformer)
        scheduler.step(valid_loss)

        end_time = time.time()
        epoch_mins, epoch_secs = epoch_time(start_time, end_time)
    
        if valid_loss < best_valid_loss:
            best_valid_loss = valid_loss
            torch.save(model.state_dict(), f'{model_name}.pt')
    
        train_history.append(train_loss)
        valid_history.append(valid_loss)
        
        print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
        print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
        print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')