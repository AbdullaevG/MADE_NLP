import utils
import imp
import torch
import tqdm
from nltk.translate.bleu_score import corpus_bleu
import random
import numpy as np

imp.reload(utils)
generate_translation = utils.generate_translation
remove_tech_tokens = utils.remove_tech_tokens
flatten = utils.flatten
get_text = utils.get_text
count_parameters = utils.count_parameters


def get_bleu_score(model, test_iterator, trg_vocab): # , bert=False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_text = []
    generated_text = []
    model.eval()
    with torch.no_grad():

        for i, batch in tqdm.tqdm(enumerate(test_iterator)):

            src = batch.src
            trg = batch.trg

            output = model(src, trg, 0)
            output = output.argmax(dim=-1)

            original_text.extend([get_text(x, trg_vocab) for x in trg.cpu().numpy().T])
            generated_text.extend([get_text(x, trg_vocab) for x in output[1:].detach().cpu().numpy().T])
            
            if i == 1:
                mini_test_trg = [get_text(x, trg_vocab) for x in trg.cpu().numpy().T]
                mini_test_gen = [get_text(x, trg_vocab) for x in output[1:].detach().cpu().numpy().T]
                
        score = corpus_bleu([[text] for text in original_text], generated_text) * 100

    return original_text, generated_text, score, mini_test_trg, mini_test_gen


def get_results(model, test_iterator, trg_vocab, tr_flag=False): # bert_flag=False, 
    print(f'The model has {count_parameters(model):,} trainable parameters')
    original_text, generated_text, score, mini_test_trg, mini_test_gen = get_bleu_score(model, test_iterator, trg_vocab) # , bert=bert_flag
    print('BLEU score:', score)
    
    scores_mini_data = []
    for i in range(len(mini_test_trg)):
        original = mini_test_trg[i]
        generated = mini_test_gen[i]
        scores_mini_data.append(corpus_bleu([original], [generated]) * 100)
    
    size = len(scores_mini_data)
    scores_mini_data = np.array(scores_mini_data)
    sorted_idx = np.argsort(scores_mini_data)
    
    print("Successful examples of translation:\n")
    for k in range(1, 4):
        print('Original:', ' '.join(mini_test_trg[sorted_idx[size-k]]))
        print('Generated:', ' '.join(mini_test_gen[sorted_idx[size-k]]))
        print()
        
    print("Bad translation examples: \n")
    for k in range(1, 4):
        print('Original:', ' '.join(mini_test_trg[sorted_idx[k]]))
        print('Generated:', ' '.join(mini_test_gen[sorted_idx[k]]))
        print()
        
        
        
        