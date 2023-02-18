from train.traincontrast import ContrastivePredictiveCoding

if __name__ == "__main__":
    tau = 0.07
    embeddingdim = 3
    contrastdim = 3
    nhead = 1
    mlpuse = False
    logpath = f"./logs/CPCtrajpairembeddim{embeddingdim}tau{tau}nhead{nhead}contrastdim{contrastdim}usemlp{mlpuse}"
    # nhead = 1
    CPC = ContrastivePredictiveCoding(path = logpath,device="cuda",usemlp = mlpuse,embeddingdim=embeddingdim,n_head=nhead,EPOCH=100,contrastdim=contrastdim,trajcompare = True,negativesamples=1024,tau=tau)
    CPC.train()