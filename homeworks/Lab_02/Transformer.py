import torch
import torch.nn as nn
import torch.optim as optim
import torchtext
import random
import math


class Encoder(nn.Module):
    def __init__(
        self,
        src_dim, 
        d_model, 
        nhead, 
        num_enc_layers, 
        dim_feedforward, 
        dropout, 
        device,
        src_pad_index,
        max_len=100
    ):
        
        super().__init__()
        
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
        
        self.src_embedding = nn.Embedding(src_dim, d_model)
        self.src_pos_enc = nn.Embedding(max_len, d_model)
        
        transformer_enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,)
        
        enc_norm = nn.LayerNorm(d_model)
        
        self.transformer_encoder = nn.TransformerEncoder(transformer_enc_layer, num_enc_layers, enc_norm)
        
        self.device = device
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.src_pad_index = src_pad_index
        
        
    def forward(self, src):
        # src = [src sent len, batch size]
        src_embedded = self.src_embedding(src) * self.scale
        # src_embedded = [src sent len, batch size, d_model]
        
        src_pos = torch.arange(0, src.shape[0]).unsqueeze(0).repeat(src.shape[1], 1)
        # src_pos = [batch size, src sent len]
        
        src_pos = src_pos.permute(1, 0).to(self.device)
        # src_pos = [src sent len, batch size]
        
        src_embedded = self.dropout(self.src_pos_enc(src_pos) + src_embedded)
        # src_embedded = [src sent len, batch size, d_model]
        
        src_padded_mask = (src.cpu() == self.src_pad_index)
        src_padded_mask = src_padded_mask.permute(1, 0).to(self.device)
        # src_padded_mask = [batch_size, src sent len]
        
        output = self.transformer_encoder(
            src_embedded, 
            src_key_padding_mask=src_padded_mask, 
        )
        # output = [src sent len, batch_size, d_model]
        
        return output
        

class Decoder(nn.Module):
    def __init__(
        self,
        trg_dim,
        output_dim,
        d_model, 
        nhead, 
        num_dec_layers, 
        dim_feedforward, 
        dropout, 
        device,
        src_pad_index,
        trg_pad_index,
        max_len=100
    ):
        
        super().__init__()
        
        self.scale = torch.sqrt(torch.FloatTensor([d_model])).to(device)
        
        self.trg_embedding = nn.Embedding(trg_dim, d_model)
        self.trg_pos_decoder = nn.Embedding(max_len, d_model)
        
        transformer_dec_layer = nn.TransformerDecoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,)
        
        dec_norm = nn.LayerNorm(d_model)
        
        # for generating mask
        self.generate_square_subsequent_mask = nn.Transformer().generate_square_subsequent_mask
        
        self.transformer_dec = nn.TransformerDecoder(transformer_dec_layer, num_dec_layers, dec_norm)
        
        self.output_layer = nn.Linear(d_model, output_dim)
        
        self.device = device
        
        self.dropout = nn.Dropout(p=dropout)
        
        self.src_pad_index = src_pad_index
        self.trg_pad_index = trg_pad_index
        
        
    def forward(self, src, memory, trg):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        # memory = [src sent len, batch_size, d_model]
        
        trg_embedded = self.trg_embedding(trg) * self.scale
        # trg_embedded = [trg sent len, batch size, d_model]
        
        trg_pos = torch.arange(0, trg.shape[0]).unsqueeze(0).repeat(trg.shape[1], 1)
        # trg_pos = [batch size, trg sent len]
        
        trg_pos = trg_pos.permute(1, 0).to(self.device)
        # trg_pos = [trg sent len, batch size]
        
        trg_embedded = self.dropout(self.trg_pos_decoder(trg_pos) + trg_embedded)
        # trg_embedded = [trg sent len, batch size, d_model]
        
        memory_padding_mask = (src.cpu() == self.src_pad_index)
        memory_padding_mask = memory_padding_mask.permute(1, 0).to(self.device)
        # memory_padding_mask = [batch_size, src_len]
        
        trg_padded_mask = (trg.cpu() == self.trg_pad_index)
        trg_padded_mask = trg_padded_mask.permute(1, 0).to(self.device)
        # trg_padded_mask = [batch_size, trg_len]
        
        trg_mask = self.generate_square_subsequent_mask(trg_embedded.size(0)).to(self.device)
        # trg_mask = [trg_len, trg_len]
        
        output = self.transformer_dec( 
            trg_embedded,
            memory,
            tgt_mask=trg_mask, 
            tgt_key_padding_mask=trg_padded_mask,
            memory_key_padding_mask=memory_padding_mask
        )
        # output = [trg_len, batch_size, d_model]
        
        output = self.output_layer(output)
        # output = [trg_len, batch_size, output_dim]
        
        return output
    

class Seq2Seq(nn.Module):
    def __init__(
        self, 
        encoder,
        decoder
    ):
        super().__init__()
        
        self.encoder = encoder
        self.decoder = decoder
        
        
    def forward(self, src, trg):
        
        # src = [src sent len, batch size]
        # trg = [trg sent len, batch size]
        memory = self.encoder(src)
        output = self.decoder(src, memory, trg)
        
        return output

HID_DIM = 512
N_HEAD = 8
N_LAYERS = 3
DIM_FEEDFORWARD = 2048
DROPOUT = 0.1
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def initialize_weights(m):
    if hasattr(m, 'weight') and m.weight.dim() > 1:
        nn.init.xavier_uniform_(m.weight.data)

def get_model(src,
              trg,
              hid_dim = HID_DIM,
              n_head = N_HEAD,
              n_layers = N_LAYERS,
              dim_feedforward = DIM_FEEDFORWARD,
              dropout = DROPOUT,
              device = DEVICE):
    
    encoder = Encoder(    
                        src_dim = len(src.vocab), 
                        d_model = hid_dim, 
                        nhead = n_head, 
                        num_enc_layers = n_layers, 
                        dim_feedforward= dim_feedforward, 
                        dropout = dropout, 
                        device = device,
                        src_pad_index = src.vocab.stoi["<pad>"]
                     )
    
    decoder = Decoder(
                        trg_dim = len(trg.vocab),
                        output_dim = len(trg.vocab),
                        d_model = hid_dim,
                        nhead = n_head, 
                        num_dec_layers = n_layers, 
                        dim_feedforward = dim_feedforward, 
                        dropout = dropout, 
                        device = device,
                        src_pad_index = src.vocab.stoi["<pad>"],
                        trg_pad_index = trg.vocab.stoi["<pad>"],
                     )
    
    model = Seq2Seq(encoder, decoder).to(device)
    model.apply(initialize_weights)
    return model
    
    
