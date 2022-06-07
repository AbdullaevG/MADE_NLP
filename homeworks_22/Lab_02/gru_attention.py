import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import pretr_emb


def emb_layer(num_embed, embedding_dim, field_vocab = None, pretr = False, eng = True):
    if pretr:
        return pretr_emb.get_emb_model(field_vocab, eng)
    else:
        return nn.Embedding(num_embed, embedding_dim)


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout,
                 field_vocab = None, pretr = False, eng=False):
        
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers
        
        self.embedding = emb_layer(
                                    num_embed = input_dim,
                                    embedding_dim = emb_dim,
                                    field_vocab = field_vocab,
                                    eng = eng,
                                    pretr = pretr
                                    )
        
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout,
            bidirectional=True
        )
        
        self.fc_out = nn.Linear(in_features=hid_dim*2, out_features=hid_dim)
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, src):
        #src = [src sent len, batch size]
        
        embedded = self.embedding(src) 
        embedded = self.dropout(embedded) 
        # embedded = [src sent len, batch size, emb dim]
        
        output, hidden = self.rnn(embedded)
        # output = [src sent len, batch_size, hidden_size*num_directions]
        # hidden = [num_layers*num_directions, batch_size, hidden_size]
        
        hidden = torch.tanh(self.fc_out(torch.cat((hidden[-2, :, :], hidden[-1, :, :]), dim=1)))
        hidden = hidden.unsqueeze(0)
        # hidden = [1, batch_size, hidden_size]
        
        return output, hidden
    
    
class Attention(nn.Module):
    def __init__(self, dec_hid_dim, enc_output_dim):
        super().__init__()
        self.dec_hid_dim = dec_hid_dim
        self.enc_output_dim = enc_output_dim
        
        self.att = nn.Linear(in_features=enc_output_dim+dec_hid_dim, 
                             out_features=dec_hid_dim)
        
        self.v = nn.Linear(in_features=dec_hid_dim,
                           out_features=1,
                           bias=False)
        
    def forward(self, hidden, enc_outputs):
        # hidden = [1, batch size, hid dim]
        
        hidden = hidden.repeat(enc_outputs.shape[0], 1, 1)
        
        output = torch.cat((hidden, enc_outputs), dim=2)
        # output = [src sent len, batch_size, hid_dim]
        
        attention = torch.tanh(self.att(output))
        attention = self.v(attention).squeeze(dim=2)
        # attention = [src_sent_len, batch_size]
        
        return nn.functional.softmax(attention, dim=0)
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout, attention,
                 field_vocab = None, eng=True, pretr = False):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.attention = attention
        
        self.embedding = emb_layer(
                                    num_embed = output_dim,
                                    embedding_dim = emb_dim,
                                    field_vocab = field_vocab,
                                    eng = eng,
                                    pretr = pretr
                                   )
        
        self.rnn = nn.GRU(
            input_size=hid_dim*2+emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        
        self.out = nn.Linear(
            in_features=hid_dim*3+emb_dim,
            out_features=output_dim
        )
        
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input, hidden, enc_outputs):
        
        # input = [batch_size]
        # hidden = [n_layers * n_directions, batch_size, hid_dim]
        # enc_outputs = [src_sent_len, batch_size, hidden_size*2]
        
        input = input.unsqueeze(0)  
        # input = [1, batch_size]
        
        embedded = self.dropout(self.embedding(input))
        # embedded = [1, batch_size, emb_dim]
        
        a = self.attention(hidden, enc_outputs)
        # a = [src sent len, batch_size]
        
        a = a.unsqueeze(1).permute(2, 1, 0)
        # a = [batch_size, 1, src sent len]
        
        enc_outputs = enc_outputs.permute(1, 0, 2)
        # enc_outputs = [batch_size, src sent len, hidden_size*2]
        
        weights = torch.bmm(a, enc_outputs)
        # weights = [batch_size, 1, hidden_size*2]
        
        weights = weights.permute(1, 0, 2)
        # weights = [1, batch_size, hidden_size*2]
        
        rnn_input = torch.cat((embedded, weights), dim=2)
        # rnn_input = [1, batch_size, hidden_size*2 + emb_size]
        
        output, hidden = self.rnn(rnn_input, hidden)
        # output = [1, batch_size, hidden_size]
        # hidden = [1, batch_size, hidden_size]
        
        output = torch.cat((output, weights, embedded), dim=2)
        # output = [1, batch_size, hidden_size + hidden_size*2 + emb_size]
        
        prediction = self.out(output.squeeze(0))   
        #prediction = [batch_size, output_dim]
        
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        #src = [src_sent_len, batch_size]
        #trg = [trg_sent_len, batch_size]
        #teacher_forcing_ratio is probability to use teacher forcing
        #e.g. if teacher_forcing_ratio is 0.75 we use ground-truth inputs 75% of the time
        
        # Again, now batch is the first dimention instead of zero
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        #tensor to store decoder outputs
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)
        
        #last hidden state of the encoder is used as the initial hidden state of the decoder
        enc_outputs, hidden = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden = self.decoder(input, hidden, enc_outputs)
            outputs[t] = output
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
        
        return outputs
    
ENC_EMB_DIM = 300
DEC_EMB_DIM = 300
HID_DIM = 512
N_LAYERS = 1
ENC_DROPOUT = 0.5
DEC_DROPOUT = 0.5

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def init_weights(m):
    for name, param in m.named_parameters():
        nn.init.uniform_(param, -0.08, 0.08)

def get_model(src,
              trg,
              pretr = False,
              enc_emb_dim = ENC_EMB_DIM,
              dec_emb_dim = DEC_EMB_DIM,
              hid_dim = HID_DIM,
              n_layers = N_LAYERS,
              enc_dropout = ENC_DROPOUT,
              dec_dropout = DEC_DROPOUT):
    
    enc = Encoder(len(src.vocab), enc_emb_dim, hid_dim, n_layers, enc_dropout,
                  field_vocab = src.vocab, pretr = pretr)
    attention = Attention(hid_dim, hid_dim * 2)
    
    dec = Decoder(len(trg.vocab), dec_emb_dim, hid_dim, n_layers, dec_dropout, attention, field_vocab = trg.vocab, pretr = pretr)
    
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)
    
    return model


