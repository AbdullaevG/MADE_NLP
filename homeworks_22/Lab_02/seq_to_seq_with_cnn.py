import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchtext




class Positional_Encoding(nn.Module):
    def __init__(self, dim, dropout=0.1, max_len=5000):
        super(Positional_Encoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pos_enc = torch.zeros(max_len, dim)
        position_matrix = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        denom = torch.exp(torch.arange(0, dim, 2).float() * (-math.log(10000.0) / dim))
        pos_enc[:, 0::2] = torch.sin(position_matrix * denom)
        pos_enc[:, 1::2] = torch.cos(position_matrix * denom)
        pos_enc = pos_enc.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pos_enc', pos_enc)

    def forward(self, x):
        out = x + self.pos_enc[:x.size(0), :]
        out = self.dropout(out)
        return out


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, dropout):
        super().__init__()
        
        self.input_dim = input_dim
        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        
        self.pos_enc = Positional_Encoding(emb_dim, dropout)
        self.embedding = nn.Embedding(input_dim,  emb_dim)
        
        self.cnn_1 = nn.Conv1d(in_channels = emb_dim, out_channels = hid_dim, kernel_size = 5) 
        self.cnn_2 = nn.Conv1d(in_channels = emb_dim, out_channels = hid_dim, kernel_size = 3)
        self.cnn_3 = nn.Conv1d(in_channels = emb_dim, out_channels = hid_dim, kernel_size = 2)
        
        self.dropout = nn.Dropout(p=dropout)
        self.relu = nn.ReLU()
        
        self.pool = nn.AdaptiveMaxPool1d(output_size=1)
        
    def forward(self, src):
        
        #src = [src_sent len, batch_size]
        src = self.embedding(src) * math.sqrt(self.emb_dim)
         #src = [src_sent len, batch_size, emb_dim]
            
        embedded = self.pos_encoder(src).permute(1, 2, 0)
        #embedded = [src_sent_len, batch_size, emb_dim]
        
        output_1 = self.cnn_1(embedded)
        output_1 = self.relu(output_1)
        #output_1 = [batch_size, hid_dim, src_sent_len_1]
        
        output_2 = self.cnn_2(embedded)
        output_2 = self.relu(output_2)
        #output2 = [batch size, hid_dim, src_sent_len_2]
        
        output_3 = self.cnn_3(embedded)
        output_3 = self.relu(output_3)
        #output3 = [batch size, hid_dim, src_sent_len_3]
        
        output = torch.cat((output_1, output_2, output_3), dim=-1)
        #output = [batch_size, hid_dim, src_sent_len_1 + src_sent_len_2 + src_sent_len_3]
        
        output = self.pool(output).permute(2, 0, 1)
        #output = [1, batch_size, hid_dim]
        
        return output
    

class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super().__init__()

        self.emb_dim = emb_dim
        self.hid_dim = hid_dim
        self.output_dim = output_dim
        self.n_layers = n_layers
        self.dropout = dropout
        
        self.embedding = nn.Embedding(output_dim, emb_dim)
        
        self.rnn = nn.GRU(
            input_size=emb_dim,
            hidden_size=hid_dim,
            num_layers=n_layers,
            dropout=dropout
        )
        
        self.out = nn.Linear(hid_dim, output_dim)
        self.dropout = nn.Dropout(p=dropout)
        
    def forward(self, input, hidden):
        
        input = input.unsqueeze(0)
        #input = [1, batch size]
        
        embedded = self.dropout(self.embedding(input)) 
        #embedded = [1, batch size, emb dim]
        
        output, hidden = self.rnn(embedded, hidden)
        prediction = self.out(output.squeeze(0))
        
        return prediction, hidden


class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder, device):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        self.device = device
        
        
    def forward(self, src, trg, teacher_forcing_ratio = 0.5):
        
        batch_size = trg.shape[1]
        max_len = trg.shape[0]
        trg_vocab_size = self.decoder.output_dim
        
        outputs = torch.zeros(max_len, batch_size, trg_vocab_size).to(self.device)

        hidden = self.encoder(src)
        
        #first input to the decoder is the <sos> tokens
        input = trg[0,:]
        
        for t in range(1, max_len):
            
            output, hidden = self.decoder(input, hidden)
            outputs[t] = output
            
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = output.max(1)[1]
            input = (trg[t] if teacher_force else top1)
            
        return outputs

ENC_EMB_DIM = 256
DEC_EMB_DIM = 256
HID_DIM = 512
N_LAYERS = 1
ENC_DROPOUT = 0.2
DEC_DROPOUT = 0.2


def init_weights(m):
    for name, param in m.named_parameters():
        if 'weight' in name:
            nn.init.normal_(param.data, mean=0, std=0.01)
        else:
            nn.init.constant_(param.data, 0)
            

def get_model(src,
              trg,
              enc_emb_dim = ENC_EMB_DIM,
              dec_emb_dim = DEC_EMB_DIM,
              hid_dim = HID_DIM,
              n_layers = N_LAYERS,
              enc_dropout = ENC_DROPOUT,
              dec_dropout = DEC_DROPOUT):
     
    inp_dim = len(src.vocab)
    out_dim = len(trg.vocab)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    enc = Encoder(inp_dim, enc_emb_dim, hid_dim, enc_dropout)
    dec = Decoder(out_dim, dec_emb_dim, hid_dim, n_layers, dec_dropout)
    model = Seq2Seq(enc, dec, device).to(device)
    model.apply(init_weights)

    return model
    
