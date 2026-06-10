import os
import torch as T
import torch.optim as optim
import torch.nn.functional as F
import torch.nn as nn
from typing import Optional
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch import Tensor
import math
'''
Positional Encoding : takes a 2d tensor --> 3d tensor
Injects some information on the relevant position of the img in the sequence
'''


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout=0.1, max_len=1024):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = T.zeros(max_len, d_model)
        position = T.arange(0, max_len, dtype=T.float).unsqueeze(1)
        div_term = T.exp(
            T.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = T.sin(position * div_term)
        if d_model % 2 == 0:
            pe[:, 1::2] = T.cos(position * div_term)
        else:
            # Slice the 1D div_term
            pe[:, 1::2] = T.cos(position * div_term[:-1])
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x, indices=None):
        # x: (batch_size, seq_len, d_model) if batch_first=True
        # self.pe is (max_len, 1, d_model)
        seq_len = x.size(1)
        
        if indices is not None:
            # Select specific encodings for each token in the sequence
            # (seq_len, 1, D)
            pe_slice = self.pe[indices, :] 
            # (1, seq_len, D)
            x = x + pe_slice.transpose(0, 1)
        else:
            # Default sequential encoding
            x = x + self.pe[:seq_len, :].transpose(0, 1)
            
        return self.dropout(x)


''' 
Recreate the transfomer layers done in the following paper
https://arxiv.org/pdf/1910.06764.pdf
'''


class TEL(TransformerEncoderLayer):
    def __init__(self,
                 d_model,
                 nhead,
                 n_layers=1,
                 dim_feedforward=256,
                 activation="relu",
                 dropout=0,
                 layer_norm_eps=1e-5,
                 batch_first=False):
        super().__init__(d_model, nhead, dim_feedforward, dropout, activation,
                         layer_norm_eps, batch_first)
        # 2 GRUs are needed - 1 for the beginning / 1 at the end
        self.gru_1 = nn.GRU(d_model,
                            d_model,
                            num_layers=n_layers,
                            batch_first=True)
        self.gru_2 = nn.GRU(input_size=d_model,
                            hidden_size=d_model,
                            num_layers=n_layers,
                            batch_first=True)

    def flatten_parameters(self):
        self.gru_1.flatten_parameters()
        self.gru_2.flatten_parameters()

    def forward(self,
                src: Tensor,
                src_mask: Optional[Tensor] = None,
                src_key_padding_mask: Optional[Tensor] = None,
                **kwargs) -> Tensor:
        self.flatten_parameters()
        h = (src).sum(dim=1).unsqueeze(dim=0)
        src = self.norm1(src)
        out, weights = self.self_attn(src,
                                    src,
                                    src,
                                    attn_mask=src_mask,
                                    key_padding_mask=src_key_padding_mask,
                                    need_weights=True)

        out, h = self.gru_1(out, h)
        out = self.norm2(out)
        out = self.activation(self.linear1(out))
        out = self.activation(self.linear2(out))
        out, h = self.gru_2(out, h)
        return out, weights


'''
Implementation of transfomer model using GRUs
'''


class GTrXL(nn.Module):
    def __init__(self,
                 d_model,
                 nheads,
                 transformer_layers,
                 hidden_dims=256,
                 n_layers=1,
                 layer_norm_eps=1e-5,
                 batch_first=False,
                 chkpt_dir="models",
                 activation='relu',
                 dropout=0.1, # Added default dropout
                 network_name='network.pt'):
        super(GTrXL, self).__init__()
        # Module layers
        self.embed = PositionalEncoding(d_model, dropout=dropout)
        encoded = TEL(d_model,
                      nheads,
                      n_layers,
                      dim_feedforward=hidden_dims,
                      activation=activation,
                      layer_norm_eps=layer_norm_eps,
                      dropout=dropout, # Pass dropout to layer
                      batch_first=batch_first)
        self.transfomer = TransformerEncoder(encoded, transformer_layers)
        self.file = os.path.join(chkpt_dir, network_name)

    def flatten_parameters(self):
        for layer in self.transfomer.layers:
            if hasattr(layer, 'flatten_parameters'):
                layer.flatten_parameters()

    def forward(self, x, indices=None):
        self.flatten_parameters()
        x = self.embed(x, indices=indices)
        # Standard TransformerEncoder.forward doesn't return weights easily if we use the default module.
        all_weights = []
        for mod in self.transfomer.layers:
            # TEL forward also needs to be updated or it will error on extra args
            x, weights = mod(x)
            all_weights.append(weights)
        return x, all_weights[-1]

    def save(self):
        T.save(self.state_dict(), self.file)

    def load(self):
        self.load_state_dict(T.load(self.file))
