import torch.nn as nn
import torch

class Stateactiontransformer(nn.Module):
    def __init__(self,
                 embeddim =32,
                 nhead = 8,
                 encoder_layer = 6,
                 batch_first = True,
                 forwarddim = 128) -> None:
        super(Stateactiontransformer,self).__init__()
        self.transformer = nn.Transformer(
            d_model = embeddim,
            nhead = nhead,
            num_encoder_layers = encoder_layer,
            num_decoder_layers = encoder_layer,
            dim_feedforward = forwarddim
        )

    def forward(self,stateactionsequence):
        device = next(self.parameters()).device
        length = stateactionsequence.shape()[0]
        mask = torch.tril((length,length)).to(device)
        return self.transformer(src = stateactionsequence,
                                tgt = stateactionsequence,
                                src_mask = mask,
                                tgt_mask = mask)[:,-1,:]

