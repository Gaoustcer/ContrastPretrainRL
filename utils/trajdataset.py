from torch.utils.data import Dataset
import numpy as np
import pickle
import torch
from tqdm import tqdm
from random import randint


class Trajectory(object):
    def __init__(self,observations,actions,sequencelength = 32):
        self.seqlen = sequencelength
        self.observations = observations
        self.actions = actions
        self.totallength = len(self.actions)
    
    def sample(self,length = None):
        if length == None:
            length = self.seqlen
        



class Trajdataset(Dataset):
    def __init__(self,
                 path = "./dataset.pkl",
                 device = "cuda"):
        with open(path,"rb") as fp:
            data = pickle.load(fp)
        keys = ['observations','actions']
        self.dataset = []
        for Traj in tqdm(data):
            traj = {}
            for key in keys:
                traj[key] = torch.from_numpy(np.stack(Traj[key],axis=0)).to(device)
            self.dataset.append(traj)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        return self.dataset[index]
    
if __name__ == "__main__":
    data = Trajdataset("../dataset.pkl",device='cpu')
    print(data[0])
    minlen = 100000
    for d in data:
        minlen = min(minlen,len(d['actions']))
    print(minlen)
