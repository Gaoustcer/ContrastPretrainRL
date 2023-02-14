from utils.infonce import infonce
from utils.trajdataset import Trajdataset,WholeTraj
from model.actionstateencoder import ActionStateencoder
from model.stateactiontransformer import Stateactiontransformer
from utils.config import *
import torch
from torch.utils.tensorboard import SummaryWriter
import d4rl
import gym
from torch.utils.data import DataLoader
from tqdm import tqdm
import itertools
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os


class ContrastivePredictiveCoding(object):
    def __init__(self,
                 path = "./logs/CPC",
                 device = "cpu",
                 embeddingdim = 32,
                 n_head = 8,
                 batch_size = 16,
                 EPOCH = 1024,
                 negativesamples = 1024,
                 tau = 0.07,
                 trajcompare = False) -> None:

        self.figure = plt.figure()
        self.ax1 = plt.axes(projection = '3d')
        
        self.negativesamples = negativesamples
        self.writer = SummaryWriter(path)
        picturepath = os.path.join(path,"distributionoftraj")
        if os.path.exists(picturepath) == False:
            os.mkdir(picturepath)
        self.picturepath = picturepath
        self.env = gym.make(envname)
        self.wholetraj = WholeTraj()
        self.actiondim = len(self.env.action_space.sample())
        self.statedim = len(self.env.observation_space.sample())
        self.lossfunction = infonce(querydim=embeddingdim,keydim=embeddingdim).to(device)
        self.Actionencoding = ActionStateencoder(
            adim=self.actiondim,
            embeddim=embeddingdim
        ).to(device)
        self.Stateencoding = ActionStateencoder(
            adim=self.statedim,
            embeddim=embeddingdim
        ).to(device)
        self.trajcompare = trajcompare
        self.dataset = Trajdataset(device=device,comparewithintraj=trajcompare)
        self.loader = DataLoader(self.dataset,batch_size=batch_size)
        self.batchsize = batch_size
        self.transformer = Stateactiontransformer(
            embeddim=embeddingdim,nhead=n_head
        ).to(device)
        self.EPOCH = EPOCH
        self.logid = 0
        self.embeddingdim = embeddingdim
        self.optimizer = torch.optim.Adam(
            itertools.chain(
                list(self.Actionencoding.parameters()),
                list(self.Stateencoding.parameters()),
                list(self.transformer.parameters()),
                list(self.lossfunction.parameters())
            ),lr = 0.0001
        )
        self.tau = tau
    
    def samplenegative(self):
        negativestates = []
        negativeactions = []
        for _ in range(self.negativesamples):
            from random import randint
            index = randint(0,len(self.wholetraj))
            s,a = self.wholetraj[index]
            negativestates.append(s)
            negativeactions.append(a)
        return torch.stack(negativestates,dim=0),torch.stack(negativeactions,dim=0)
    
    def visualize(self,index):
        for states,actions,_,_ in tqdm(self.loader):
            statesembedding = self.Stateencoding(states)
            actionemebdding = self.Actionencoding(actions)
            sequenceembedding = self.combinestatesactions(
                statesembedding,actionemebdding)
            sequencetrans = self.transformer(sequenceembedding)
        picture = os.path.join(self.picture)

    def train(self):
        for epoch in range(self.EPOCH):
            if self.trajcompare == False:
                self.trainanapoch()
            else:
                self.traintrajectory()
    def combinestatesactions(self,s_,a_):
        return torch.concat((s_,a_),dim=-1).reshape(s_.shape[0],2 * s_.shape[1],-1)
    def traintrajectory(self):
        s_,a_ = self.samplenegative()
        s_ = self.Stateencoding(s_)
        a_ = self.Actionencoding(a_)
        negativesequence = self.combinestatesactions(s_,a_).detach()
        for states,actions,positivestates,positiveactions in tqdm(self.loader):
            self.optimizer.zero_grad()
            statesembedding = self.Stateencoding(states)
            actionembedding = self.Actionencoding(actions)
            positiveactionembedding = self.Actionencoding(positiveactions)
            positivestateembedding = self.Stateencoding(positivestates)
            sequence = self.combinestatesactions(statesembedding,actionembedding)
            positivesequence = self.combinestatesactions(positivestateembedding,positiveactionembedding)
            
            sequenceembedding = self.transformer(sequence)
            positivesequenceembedding = self.transformer(positivesequence)
            negativesequenceembedding = self.transformer(negativesequence).detach()
            loss = self.lossfunction.forward(sequenceembedding,positivesequenceembedding,negativesequenceembedding)
            loss = torch.mean(loss)
            loss.backward()
            self.writer.add_scalar("loss",loss,self.logid)
            self.logid += 1
            self.optimizer.step()

    def trainanapoch(self):
        for states,actions,actionssamples in tqdm(self.loader):
            print(states.shape,actions.shape,actionssamples.shape)
            self.optimizer.zero_grad()
            statesembedding = self.Stateencoding(states)
            actionsembedding = self.Actionencoding(actions)
            negativesamples = self.Actionencoding(actionssamples).detach().reshape(
                -1,self.embeddingdim)
            '''
            generate sequence data
            s_1,s_2,\cdots,s_t + a_1,a_2,\cdots,a_t = s_1,a_1,s_2,a_2,\cdots,s_t,a_t
            '''
            sequencedata = torch.concat(
                (statesembedding,actionsembedding),
                dim=-1
            ).reshape(statesembedding.shape[0],-1,self.embeddingdim)[:,:-1,:]
            sequenceembedding = self.transformer(sequencedata)
            '''
            sequenceembedding is [batch_size,embed_dim]
            '''
            positivesamples = actionsembedding[:,-1,:]
            '''
            positive data is [batch_size,embed_dim]
            '''
            loss = torch.mean(self.lossfunction.forward(query = sequenceembedding,
                                                        positivekeys = positivesamples,
                                                        negativekeys = negativesamples))
            loss.backward()
            self.optimizer.step()
            self.writer.add_scalar("loss",loss,self.logid)
            self.logid += 1

    
        
        pass