from utils.trajdataset import Trajdataset
from train.train import ContrastivePredictiveCoding
if __name__ == "__main__":
    # CPC = ContrastivePredictiveCoding()
    # print("Create Over")
    CPC = ContrastivePredictiveCoding(device='cuda')
    CPC.train()
    # dataset = Trajdataset(device="cuda")
    # for traj in dataset:
    #     print(traj.observations.shape,traj.actions.shape)