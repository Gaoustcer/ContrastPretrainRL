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

    def sampleactions(self):
        index = np.random.randint(0,self.totallength,(self.seqlen,))
        return self.actions[index]

    def sample(self,length = None):
        if length == None:
            length = self.seqlen
        assert length < self.totallength/2
        startentry = randint(0,self.totallength - 2 * length)
        '''
        sample s_0,s_1,s_2,\cdots,s_t a_0,a_1,a_2,\cdots,a_{t-1}, this is transition
        positive actions a_t
        '''
        return self.observations[startentry:startentry + length,:],\
            self.actions[startentry:startentry + length,:],\
            self.sampleactions()


class Trajdataset(Dataset):
    def __init__(self,
                 samplelength = 32,
                 path = "./dataset.pkl",
                 device = "cuda"):
        with open(path,"rb") as fp:
            data = pickle.load(fp)
        keys = ['observations','actions']
        self.dataset = []
        for Traj in tqdm(data):
            traj = {}
            traj = Trajectory(observations=torch.from_numpy(np.stack(Traj[keys[0]],axis=0)).to(device),
                              actions=torch.from_numpy(np.stack(Traj[keys[1]],axis=0)).to(device),sequencelength=samplelength)
            # for key in keys:
                # traj[key] = torch.from_numpy(np.stack(Traj[key],axis=0)).to(device)
            self.dataset.append(traj)
    
    def __len__(self):
        return len(self.dataset)

    def __getitem__(self,index):
        return self.dataset[index].sample()
    
if __name__ == "__main__":
    data = Trajdataset(path="../dataset.pkl",device='cpu')
    from torch.utils.data import DataLoader
    loader = DataLoader(data,batch_size=16)
    for instances in loader:
        print(type(instances))
        # print(instances[0].shape)
        for instance in instances:
            print(instance.shape)
        exit()
    # print(data[0]['actions'].shape,data[0]['observations'].shape)
    print(data[0].observations.shape)
    traj = data[0]
    obs,act = traj.sample(40)
    print(obs.shape)
    print(act.shape)
    # minlen = 100000
#    for d in data:
        # minlen = min(minlen,len(d['actions']))
# ?    print(minlen)
