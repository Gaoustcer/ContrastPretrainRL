from model.MLP import MultiMLPblock
from model.actionstateencoder import ActionStateencoder
from model.stateactiontransformer import Stateactiontransformer
import torch.nn as nn
import torch

class Siam(nn.Module):
    def __init__(self,actiondim,statedim,sequencelength = 32,embeddim = 32,nhead = 8,transformerembed = 128,encoder_layernum = 6,transformerforwarddim = 1024) -> None:
        super(Siam,self).__init__()
        self.embedding = nn.Embedding(2 * sequencelength,embedding_dim = transformerembed)
        self.embedMLP = nn.Sequential(
            nn.Linear(transformerembed,32),
            nn.ReLU(),
            nn.Linear(32,transformerembed)
        )
        # self.embedMLP = MultiMLPblock(input_dim = transformerembed,output_dim = transformerembed,middle_neurons = [128,256])
        self.Actionencoder = ActionStateencoder(adim = actiondim,embeddim = transformerembed)
        self.Stateencoder = ActionStateencoder(adim = statedim,embeddim = transformerembed)
        self.transformer = Stateactiontransformer(embeddim = 2 * transformerembed,nhead = nhead,batch_first = True,encoder_layer = encoder_layernum,forwarddim = transformerforwarddim)
        self.projector = MultiMLPblock(input_dim = 2* transformerembed,output_dim = embeddim,middle_neurons = [128,64])
        self.predictor = MultiMLPblock(input_dim = embeddim,output_dim = embeddim,middle_neurons = [32,16]) 
        self.transformerembedding = transformerembed
    
    def forward(self,states:torch.Tensor,actions:torch.Tensor):
        statesembedding = self.Stateencoder(states)
        actionsembedding = self.Actionencoder(actions)
        if states.dim() == 2:
            sequencenum = 1
            sequencelength = 2 * states.shape[0]
        else:
            sequencenum = states.shape[0]
            sequencelength = 2 * states.shape[1]
        index = torch.arange(sequencelength)\
            .repeat(sequencenum)\
            .reshape(sequencenum,sequencelength)\
            .to(states.device)
        # print(self.embedding(index).shape)
        embedindex = self.embedMLP(self.embedding(index))
        # print(embedindex.shape)
        '''
        embeddindex is [N,2 * L,E]
        '''
        sequencedata = torch.reshape(
            torch.concat([statesembedding,actionsembedding],dim = -1),(sequencenum,-1,self.transformerembedding)
        )
        '''
        sequencedata is [N,2 * L,E]
        '''
        # print(sequencedata.shape)
        sequenceembedding = self.transformer(torch.concat((sequencedata,embedindex),dim = -1))
        projection = self.projector(sequenceembedding)
        prediction = self.predictor(projection)
        '''
        input [N,L,E] N is the number of sequence
        sequenceembedding [N,transformerembed]
        projection [N,embed]
        prediction [N,embed]
        '''
        return sequenceembedding,projection,prediction
    

