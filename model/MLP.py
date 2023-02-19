import torch.nn as nn
import torch

class MLPblock(nn.Module):
    def __init__(self,input_dim,output_dim) -> None:
        super(MLPblock,self).__init__()
        self.MLP = nn.Sequential(
            nn.Linear(input_dim,output_dim),
            nn.BatchNorm1d(output_dim),
            nn.ReLU(),
            nn.BatchNorm1d(output_dim)
        )
        self.apply(MLPinit)
    def forward(self,inputs):
        return self.MLP(inputs)

def MLPinit(layer:nn.Linear):
    if type(layer) == nn.Linear:
        nn.init.kaiming_uniform_(layer.weight,a = -0.1,b = 0.1)

class MultiMLPblock(nn.Module):
    def __init__(self,input_dim,output_dim,middle_neurons:list) -> None:
        super(MultiMLPblock,self).__init__()
        indim = input_dim 
        middle_neurons.append(output_dim)
        self.sequencenet = nn.Sequential()
        for index,outdim in enumerate(middle_neurons):
            self.sequencenet.add_module(
                "layer{}".format(index),
                MLPblock(input_dim = indim,output_dim = outdim)
            )
            indim = outdim
    def forward(self,inputs):
        return self.sequencenet(inputs)
if __name__ == "__main__":
    mlp = MLPblock(32,64)
    tensor = torch.rand(12,32)
    print(mlp(tensor).shape)