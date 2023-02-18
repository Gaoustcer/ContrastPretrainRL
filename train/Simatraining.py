import torch
from torch.optim import Adam
from torch.utils.data import DataLoader
from utils.config import *
from utils.trajdataset import Trajdataset
from tqdm import tqdm
from model.Siammodel import Siam
from torch.utils.tensorboard import SummaryWriter
import d4rl
import gym
import torch.nn.functional as F

class Siamtrain(object):
    def __init__(self,path = "./log/Siam") -> None:
        self.EPOCH = EPOCH
        self.env = gym.make(envname)
        adim = len(self.env.action_space.sample())
        sdim = len(self.env.observation_space.sample()) 
        self.Siam = Siam(actiondim=adim,statedim=sdim).cuda()        
        self.optimizer = Adam(self.Siam.parameters(),lr = lr)
        dataset = Trajdataset(comparewithintraj = True,device = device)
        self.data = DataLoader(dataset,batch_size = Batchsize)
        self.path = path
        self.writer = SummaryWriter(self.path)
        self.logid = 0
    
    def train(self):
        for epoch in range(self.EPOCH):
            self.trainanepoch()
    def trainanepoch(self):
        for states,actions,enhancestates,enhanceactions in tqdm(self.data):
            self.optimizer.zero_grad()
            _,projection,prediction = self.Siam.forward(states,actions)
            _,enhanceprojection,enhanceprediction = self.Siam.forward(enhancestates,enhanceactions)
            enhanceprediction = enhanceprediction.detach()
            enhanceprojection = enhanceprojection.detach()
            loss = F.mse_loss(projection,enhanceprediction) \
                + F.mse_loss(enhanceprojection,prediction)
            loss.backward()
            self.writer.add_scalar("loss",loss,self.logid)
            self.logid += 1
            self.optimizer.step()