from model.MLP import MultiMLPblock
from model.actionstateencoder import ActionStateencoder
from model.stateactiontransformer import Stateactiontransformer
import torch.nn as nn
import torch

class Siam(nn.Module):
    def __init__(self,actiondim,statedim,embeddim = 32,nhead = 8,transformerembed = 128,encoder_layernum = 6,transformerforwarddim = 1024) -> None:
        super(Siam,self).__init__()
        self.Actionencoder = ActionStateencoder(adim = actiondim,embeddim = transformerembed)
        self.Stateencoder = ActionStateencoder(adim = statedim,embeddim = transformerembed)
        self.transformer = Stateactiontransformer(embeddim = transformerembed,nhead = nhead,batch_first = True,encoder_layer = encoder_layernum,forwarddim = transformerforwarddim)
        self.projector = MultiMLPblock(input_dim = transformerembed,output_dim = embeddim,middle_neurons = [128,64])
        self.predictor = MultiMLPblock(input_dim = embeddim,output_dim = embeddim,middle_neurons = [32,16]) 
        self.transformerembedding = transformerembed
    
    def forward(self,states,actions):
        statesembedding = self.Stateencoder(states)
        actionsembedding = self.Actionencoder(actions)
        if states.dim() == 2:
            sequencenum = 1
        else:
            sequencenum = states.shape[0]
        sequenceembedding = self.transformer(torch.reshape(
            torch.concat([statesembedding,actionsembedding],dim = -1),(sequencenum,-1,self.transformerembedding)
        ))
        projection = self.projector(sequenceembedding)
        prediction = self.predictor(projection)
        '''
        input [N,L,E] N is the number of sequence
        sequenceembedding [N,transformerembed]
        projection [N,embed]
        prediction [N,embed]
        '''
        return sequenceembedding,projection,prediction
    

