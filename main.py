from train.traincontrast import ContrastivePredictiveCoding
from model.Siammodel import Siam
import torch
from train.Simatraining import Siamtrain
def validateSiam():
    adim = 3
    sdim = 11
    N = 17
    T = 19
    siam = Siam(actiondim = adim,statedim = sdim)
    states = torch.rand(N,T,sdim)
    actions = torch.rand(N,T,adim)
    proj, pred = siam.forward(states,actions)
    print(proj.shape)
    print(pred.shape)

if __name__ == "__main__":
    # validateSiam()
    embeddim = 32
    nhead = 8
    transformerembed = 128
    transformerembeddim = 1024
    path = f'./log/Siam_nhead={nhead}\
        _embeddim={embeddim}\
        _transformerembed={transformerembed}\
        _transformerembeddim={transformerembeddim}'
    trainagent = Siamtrain(path = path,embeddim = embeddim,nhead = nhead,transformerembed = transformerembed,transformerforwarddim = transformerembeddim)
    trainagent.train()
# if __name__ == "__main__":
#     tau = 0.07
#     embeddingdim = 3
#     contrastdim = 3
#     nhead = 1
#     mlpuse = False
#     logpath = f"./logs/CPCtrajpairembeddim{embeddingdim}tau{tau}nhead{nhead}contrastdim{contrastdim}usemlp{mlpuse}"
#     # nhead = 1
#     CPC = ContrastivePredictiveCoding(path = logpath,device="cuda",usemlp = mlpuse,embeddingdim=embeddingdim,n_head=nhead,EPOCH=100,contrastdim=contrastdim,trajcompare = True,negativesamples=1024,tau=tau)
#     CPC.train()