from train.train import ContrastivePredictiveCoding

if __name__ == "__main__":
    tau = 0.07
    embeddingdim = 128
    contrastdim = 3
    nhead = 8
    logpath = f"./logs/CPCtrajpairembeddim{embeddingdim}tau{tau}nhead{nhead}contrastdim{contrastdim}"
    # nhead = 1
    CPC = ContrastivePredictiveCoding(path = logpath,device="cuda",embeddingdim=embeddingdim,n_head=nhead,EPOCH=20,contrastdim=contrastdim,trajcompare = True,negativesamples=1024,tau=tau)
    CPC.train()