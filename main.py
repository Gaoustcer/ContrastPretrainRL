from utils.trajdataset import Trajdataset
from train.train import ContrastivePredictiveCoding
if __name__ == "__main__":
    # CPC = ContrastivePredictiveCoding()
    # print("Create Over")
    CPC = ContrastivePredictiveCoding(path = "./logs/CPCtrajpair3",device='cuda',EPOCH=1024 * 32,trajcompare=True,n_head=1,embeddingdim=3)
    CPC.train()
    # dataset = Trajdataset(device="cuda")
    # for traj in dataset:
    #     print(traj.observations.shape,traj.actions.shape)