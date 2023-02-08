import torch.nn as nn

import torch
import numpy as np

numbersofneutral = [0,16,32,16,32,0]
class ActionStateencoder(nn.Module):
    def __init__(self,adim,embeddim = 32) -> None:
        super(ActionStateencoder,self).__init__()
        self.actionencoder = nn.Sequential()
        self.actiondim = adim
        self.embeddim = embeddim
        numbersofneutral[0] = self.actiondim
        numbersofneutral[-1] = embeddim
        for i in range(len(numbersofneutral) - 2):
            self.actionencoder.add_module(
                "encoderlayer{}".format(i),
                nn.Sequential(
                    nn.Linear(numbersofneutral[i],numbersofneutral[i+1]),
                    nn.ReLU()
                )
            )
        self.actionencoder.add_module(
            "lastlayer",
            nn.Linear(numbersofneutral[-2],numbersofneutral[-1])
        )
    
    def forward(self,actionsorstates):
        return self.actionencoder(actionsorstates)

if __name__ == "__main__":
    adim = 6
    actions = torch.rand(11,adim)
    network = ActionStateencoder(adim)
    print(network(actions).shape)
    