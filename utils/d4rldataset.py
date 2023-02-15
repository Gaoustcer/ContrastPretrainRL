import d4rl
import gym

from torch.utils.data import Dataset
from trajdataset import Trajectory

class d4rldataset(Dataset):
    def __init__(self,envname = "hopper-medium-v2",device = 'cpu'):
        env = gym.make(envname)
        data = d4rl.qlearning_dataset(env)
        keys = ['observations','actions']
        obs = data[keys[0]]
        actions = data[keys[1]]
        done = data['terminals']