import utils
import imp
import torch
import tqdm
from nltk.translate.bleu_score import corpus_bleu
import random
import numpy as np
import time

imp.reload(utils)
generate_translation = utils.generate_translation
remove_tech_tokens = utils.remove_tech_tokens
flatten = utils.flatten
get_text = utils.get_text
count_parameters = utils.count_parameters
epoch_time = utils.epoch_time
translate_sentence = utils.translate_sentence





def get_bleu(model, test_iterator, trg_vocab, batch_size):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_text = []
    generated_text = []
    time_history = []
    model.eval()
    with torch.no_grad():

        for i, batch in tqdm.tqdm(enumerate(test_iterator)):


            src = batch.src
            trg = batch.trg
            start_time = time.time()
            output = model(src, trg, 0)
            output = output.argmax(dim=-1)
            end_time = time.time()
            epoch_mins, epoch_secs = epoch_time(start_time, end_time)
            time_history.append(epoch_secs)
            
            original_text.extend([get_text(x, trg_vocab) for x in trg.cpu().numpy().T])
            generated_text.extend([get_text(x, trg_vocab) for x in output[1:].detach().cpu().numpy().T])

        score = corpus_bleu([[text] for text in original_text], generated_text) * 100
        
    return original_text, generated_text, time_history, score


    
def get_bleu_transformer(model, test_iterator, src_vocab, trg_vocab, batch_size, transformer = False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_text = []
    generated_text = []
    time_history = []
    mini_test_trg = []
    mini_test_gen = []
    model.eval()
    
    with torch.no_grad():

        for i, batch in tqdm.tqdm(enumerate(test_iterator)):


            batch_src = batch.src.permute(1, 0)
            batch_trg = batch.trg.permute(1, 0)

            for src, trg in zip(batch_src, batch_trg):
                src = src.unsqueeze(1)
                trg = trg.unsqueeze(1)
            
                start_time = time.time()
                output = translate_sentence(model, src, src_vocab, trg_vocab, device)
                end_time = time.time()
                epoch_mins, epoch_secs = epoch_time(start_time, end_time)
                time_history.append(epoch_secs)

                original_text.extend([get_text(x, trg_vocab) for x in trg[1:, :].cpu().numpy().T])
                generated_text.extend([get_text(x, trg_vocab) for x in np.array([output])])
                
        score = corpus_bleu([[text] for text in original_text], generated_text) * 100
        
    return original_text, generated_text, time_history, score

def show_results(model, test_iterator, src_vocab, trg_vocab, batch_size, transformer = False):
    print(f'The model has {count_parameters(model):,} trainable parameters')
    
    if transformer:
        original_text, generated_text, time_history, score = get_bleu_transformer( model, test_iterator, src_vocab, trg_vocab, batch_size)
        print('BLEU score:', score)
        print(f"Время на inference  в пересчете на один батч размером 32: {np.mean(time_history) * (32 / 1)} seconds")
    else:
        original_text, generated_text, time_history, score = get_bleu(model, test_iterator, trg_vocab, batch_size) 
        print('BLEU score:', score)
        print(f"Время на inference в пересчете на один батч размером 32: {np.mean(time_history) * (32 / batch_size)} seconds")
    
    scores_mini_data = []
        
    test_size = len(original_text)
    idx = set(np.random.randint(0, test_size - 1, size = 500))
    size = len(idx)
    for i in idx:
        original = original_text[i]
        generated = generated_text[i]
        scores_mini_data.append(corpus_bleu([[original]], [generated]) * 100)

    scores_mini_data = np.array(scores_mini_data)
    sorted_idx = np.argsort(scores_mini_data)
    
    print()
    print("Successful examples of translation:\n")
    for k in range(1, 4):
        print("\t", 'Original:', ' '.join(original_text[sorted_idx[size-k]]))
        print("\t", 'Generated:', ' '.join(generated_text[sorted_idx[size-k]]))
        print()
        
    print("Bad translation examples: \n")
    for k in range(1, 4): 
        print("\t", 'Original:', ' '.join(original_text[sorted_idx[k]]))
        print("\t", 'Generated:', ' '.join(generated_text[sorted_idx[k]]))
        print()