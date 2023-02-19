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
    def __init__(self,path = "./log/Siam",embeddim = 32
                 ,nhead = 8,transformerembed = 128,
                 transformerforwarddim = 1024) -> None:
        '''
        [N,L,3] + [N,L,11] 
        -> [N,2L,transformerembed] 
        -> [N,transformerembed](sequence embedding) 
        -> [N,embed](projection)
        -> [N,embed](prediction)
        '''
        self.EPOCH = EPOCH
        self.env = gym.make(envname)
        adim = len(self.env.action_space.sample())
        sdim = len(self.env.observation_space.sample()) 
        self.Siam = Siam(actiondim=adim,
                         statedim=sdim,
                         embeddim = embeddim,
                         nhead = nhead,
                         transformerembed = transformerembed,
                         transformerforwarddim = transformerforwarddim).cuda()        
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
        if os.path.exists(picroot) == False:
            os.mkdir(picroot)

        for states,actions,_,_ in tqdm(self.data):
            _,projection,prediction = self.Siam.forward(states,actions)
            projection = projection.detach().to("cpu").numpy()
            prediction = prediction.detach().to("cpu").numpy()
            self.ax.scatter(projection[:,0],projection[:,1],projection[:,2],c = 'r',s = 0.1)
            self.ax.scatter(prediction[:,0],prediction[:,1],prediction[:,2],c = 'g',s = 0.1)
        picroot = os.path.join(picroot,"{}.png".format(index))
        self.fig.savefig(picroot)
        plt.cla()
            
    def train(self):
        self.validate(-1)
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