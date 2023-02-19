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
from matplotlib import pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


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
        self.fig = plt.figure()
        self.ax = plt.axes(projection = "3d")

    def save(self):
        # import os
        modelpath = os.path.join(self.path,"model")
        if os.path.exists(modelpath) == False:
            os.mkdir(modelpath)
        modelpath = os.path.join(modelpath,"siam")
        torch.save(self.Siam,modelpath)
    
    def validate(self,index):
        picroot = os.path.join(self.path,'picture')

        for states,actions,_,_ in tqdm(self.data):
            _,projection,prediction = self.Siam.forward(states,actions)
            self.ax.scatter(x = projection[:,0],y = projection[:,1],z = projection[:,2],c = 'r',s = 0.1)
            self.ax.scatter(x = prediction[:,0],y = prediction[:,1],z = prediction[:,2],c = 'g',s = 0.1)
        picroot = os.path.join(picroot,"{}.png".format(index))
        self.fig.savefig(picroot)
        plt.cla()
            
    def train(self):
        for epoch in range(self.EPOCH):
            self.validate(epoch)
            self.trainanepoch()
        self.save()
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