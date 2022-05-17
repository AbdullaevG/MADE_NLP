import random
import torch
from torch import nn
from Attention import Attention


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(
            num_embeddings=input_dim,
            embedding_dim=emb_dim
        )
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        embedded = self.embedding(src)
        embedded = self.dropout(embedded)
        output, hidden = self.rnn(embedded)
        return output, hidden


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.embedding = nn.Embedding(num_embeddings=output_dim,
                                      embedding_dim=emb_dim)
        self.rnn = nn.LSTM(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        self.attention = Attention(hid_dim)
        self.out = nn.Linear(hid_dim*2, output_dim)        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input, hidden, enc_seq):
        input = input.unsqueeze(0)
        embedded = self.dropout(self.embedding(input))
        
        output, hidden = self.rnn(embedded, hidden)
        
        attention_output = self.attention(output.transpose(0, 1),
                                          enc_seq.transpose(0, 1))[0]
        attention_output = attention_output.transpose(0, 1)
        
        prediction = self.out((torch.cat([attention_output.squeeze(0), output.squeeze(0)], dim=1)))
        
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        assert encoder.hid_dim == decoder.hid_dim, \
            "Hidden dimensions of encoder and decoder must be equal!"
        assert encoder.n_layers == decoder.n_layers, \
            "Encoder and decoder must have equal number of layers!"
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        enc_seq, hidden = self.encoder(src)
        input = trg[0,:]
        for t in range(1, max_len):
            output, hidden = self.decoder(input, hidden, enc_seq)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs

    
def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            
ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
ENC_HID_DIM = 512
DEC_HID_DIM = 512
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5
N_LAYERS = 2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def seq_to_seq_attent_LSTM(inp_dim,
                           out_dim,
                           enc_emb_dim = ENC_EMB_DIM,
                           dec_emb_dim = DEC_EMB_DIM,
                           enc_hid_dim = ENC_HID_DIM,
                           dec_hid_dim = DEC_HID_DIM,
                           n_layers = N_LAYERS,
                           enc_dropout = ENC_DROPOUT,
                           dec_dropout = DEC_DROPOUT):
    
    enc = Encoder(inp_dim, enc_emb_dim, enc_hid_dim, n_layers, enc_dropout)
    dec = Decoder(out_dim, dec_emb_dim, dec_hid_dim, n_layers, dec_dropout)
    model = Seq2Seq(enc, dec, device)
    model.apply(init_weights)
    return model   