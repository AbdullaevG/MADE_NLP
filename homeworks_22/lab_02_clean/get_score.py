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


def translate_sentence_vectorized(model, src_tensor, trg_field, device, max_len=50):
    assert isinstance(src_tensor, torch.Tensor)

    model.eval()
    src_mask = model.make_src_mask(src_tensor)

    with torch.no_grad():
        enc_src = model.encoder(src_tensor, src_mask)
    # enc_src = [batch_sz, src_len, hid_dim]

    trg_indexes = [[trg_field.vocab.stoi[trg_field.init_token]] for _ in range(len(src_tensor))]
    # Even though some examples might have been completed by producing a <eos> token
    # we still need to feed them through the model because other are not yet finished
    # and all examples act as a batch. Once every single sentence prediction encounters
    # <eos> token, then we can stop predicting.
    translations_done = [0] * len(src_tensor)
    for i in range(max_len):
        trg_tensor = torch.LongTensor(trg_indexes).to(device)
        trg_mask = model.make_trg_mask(trg_tensor)
        with torch.no_grad():
            output, attention = model.decoder(trg_tensor, enc_src, trg_mask, src_mask)
        pred_tokens = output.argmax(2)[:,-1]
        for i, pred_token_i in enumerate(pred_tokens):
            trg_indexes[i].append(pred_token_i)
            if pred_token_i == trg_field.vocab.stoi[trg_field.eos_token]:
                translations_done[i] = 1
        if all(translations_done):
            break

    # Iterate through each predicted example one by one;
    # Cut-off the portion including the after the <eos> token
    pred_sentences = []
    for trg_sentence in trg_indexes:
        pred_sentence = []
        for i in range(1, len(trg_sentence)):
            if trg_sentence[i] == trg_field.vocab.stoi[trg_field.eos_token]:
                break
            pred_sentence.append(trg_field.vocab.itos[trg_sentence[i]])
        pred_sentence = remove_tech_tokens(pred_sentence)
        pred_sentences.append(pred_sentence)

    return pred_sentences, pred_tokens


def get_bleu_score(model, test_iterator, TRG, transformer = False): # , bert=False
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    original_text = []
    generated_text = []
    model.eval()
    with torch.no_grad():

        for i, batch in tqdm.tqdm(enumerate(test_iterator)):

            src = batch.src
            trg = batch.trg
            
            if transformer:
                translation, output = translate_sentence_vectorized(model, src, TRG, device)
                generated_text.extend(translation)
                original_text.extend([get_text(x, TRG.vocab) for x in trg])
            
            else:
                output = model(src, trg, 0)
                output = output.argmax(dim=-1)

                original_text.extend([get_text(x, TRG.vocab) for x in trg.cpu().numpy().T])
                generated_text.extend([get_text(x, TRG.vocab) for x in output[1:].detach().cpu().numpy().T])
            
            rand_batch = random.randint(0, 10)
            if i == rand_batch:
                mini_test_trg = [get_text(x, TRG.vocab) for x in trg.cpu().numpy().T]
                mini_test_gen = [get_text(x, TRG.vocab) for x in output[1:].detach().cpu().numpy().T]
                
        score = corpus_bleu([[text] for text in original_text], generated_text) * 100

    return original_text, generated_text, score, mini_test_trg, mini_test_gen


def get_results(model, test_iterator, TRG, transformer = False):
    print(f'The model has {count_parameters(model):,} trainable parameters')
    original_text, generated_text, score, mini_test_trg, mini_test_gen = get_bleu_score(model, test_iterator, TRG, transformer) # , bert=bert_flag
    print('BLEU score:', score)
    
    scores_mini_data = []
    for i in range(len(mini_test_trg)):
        original = mini_test_trg[i]
        generated = mini_test_gen[i]
        scores_mini_data.append(corpus_bleu([original], [generated]) * 100)
    
    size = len(scores_mini_data)
    scores_mini_data = np.array(scores_mini_data)
    sorted_idx = np.argsort(scores_mini_data)
    
    print()
    print("Successful examples of translation:\n")
    for k in range(1, 4):
        print("\t", 'Original:', ' '.join(mini_test_trg[sorted_idx[size-k]]))
        print("\t", 'Generated:', ' '.join(mini_test_gen[sorted_idx[size-k]]))
        print()
        
    print("Bad translation examples: \n")
    for k in range(1, 4):
        print("\t", 'Original:', ' '.join(mini_test_trg[sorted_idx[k]]))
        print("\t", 'Generated:', ' '.join(mini_test_gen[sorted_idx[k]]))
        print()
        
        
        
        