import torch
import torch.nn as nn
import numpy as np

import gensim.downloader as api
from navec import Navec

# !pip install navec
# https://github.com/natasha/navec
# wget https://storage.yandexcloud.net/natasha-navec/packs/navec_hudlit_v1_12B_500K_300d_100q.tar


def get_emb_model(field_vocab,
                  eng = True,
                  emb_dim = 300,
                  path = 'navec_hudlit_v1_12B_500K_300d_100q.tar', 
                  pretr_model_name_eng = "glove-wiki-gigaword-300", 
                  non_trainable = False):
    if eng:
        model = api.load(pretr_model_name_eng)
    else:
        model = Navec.load(path)
    
    vocab = {}
    for i in range(len(field_vocab)):
        vocab[i] = field_vocab.itos[i]
    matrix_len = len(vocab)
    weights_matrix = np.zeros((matrix_len, emb_dim))
    
    for i, word in enumerate(vocab):
        try: 
            weights_matrix[i] = model[str(word)]
        except KeyError:
            weights_matrix[i] = np.random.normal(scale=0.6, size=(emb_dim, ))
            
    weights_matrix = torch.tensor(weights_matrix, dtype = torch.float)
    num_embeddings, embedding_dim = weights_matrix.size()
    
    emb_layer = nn.Embedding(num_embeddings, embedding_dim)
    emb_layer.load_state_dict({'weight': weights_matrix})
    
    if non_trainable:
        emb_layer.weight.requires_grad = False

    return emb_layer