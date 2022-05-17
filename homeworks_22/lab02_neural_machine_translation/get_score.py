import utils
import imp
import torch
import tqdm
from nltk.translate.bleu_score import corpus_bleu
import random

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
        score = corpus_bleu([[text] for text in original_text], generated_text) * 100

    return original_text, generated_text, score


def get_results(model, test_iterator, trg_vocab, tr_flag=False, numb_of_examples=10): # bert_flag=False, 
    print(f'The model has {count_parameters(model):,} trainable parameters')
    original_text, generated_text, score = get_bleu_score(model, test_iterator, trg_vocab) # , bert=bert_flag
    print('BLEU score:', score)
    sentences_num = len(original_text)
    for _ in range(numb_of_examples):
        index = random.randint(0, sentences_num-1)
        print('Original:', ' '.join(original_text[index]))
        print('Generated:', ' '.join(generated_text[index]), "\n")
                