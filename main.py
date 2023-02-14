from train.train import ContrastivePredictiveCoding

if __name__ == "__main__":
    tau = 0.07
    embeddingdim = 3
    nhead = 1
    logpath = f"./logs/CPCtrajpairembeddim{embeddingdim}tau{tau}nhead{nhead}"
    # nhead = 1
    CPC = ContrastivePredictiveCoding(path = logpath,device="cuda",embeddingdim=embeddingdim,n_head=1,EPOCH=200,trajcompare = True,negativesamples=1024,tau=tau)
    CPC.train()