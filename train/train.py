from utils.infonce import infonce
from utils.trajdataset import Trajdataset
from model.actionstateencoder import ActionStateencoder
from model.stateactiontransformer import Stateactiontransformer
from utils.config import *
import torch
from torch.utils.tensorboard import SummaryWriter
import d4rl
import gym
from torch.utils.data import DataLoader
from tqdm import tqdm

class ContrastivePredictiveCoding(object):
    def __init__(self,
                 path = "./logs/CPC",
                 device = "cpu",
                 embeddingdim = 32,
                 batch_size = 16,
                 EPOCH = 1024) -> None:
        self.writer = SummaryWriter(path)
        self.env = gym.make(envname)
        self.actiondim = len(self.env.action_space.sample())
        self.statedim = len(self.env.observation_space.sample())
        self.Actionencoding = ActionStateencoder(
            adim=self.actiondim,
            embeddim=embeddingdim
        ).to(device)
        self.Stateencoding = ActionStateencoder(
            adim=self.statedim,
            embeddim=embeddingdim
        ).to(device)
        self.dataset = Trajdataset(device=device)
        self.loader = DataLoader(self.dataset,batch_size=batch_size)
        self.transformer = Stateactiontransformer(
            embeddim=embeddingdim
        ).to(device)
        self.EPOCH = EPOCH
        self.logid = 0
        self.embeddingdim = embeddingdim
    
    def train(self):
        for epoch in range(self.EPOCH):
            self.trainanapoch()
    def trainanapoch(self):
        for states,actions,actionssamples in tqdm(self.loader):
            statesembedding = self.Stateencoding(states)
            actionsembedding = self.Actionencoding(actions)
            negativesamples = self.Actionencoding(actionssamples).detach().reshape(
                -1,self.embeddingdim)
            '''
            generate sequence data
            s_1,s_2,\cdots,s_t + a_1,a_2,\cdots,a_t = s_1,a_1,s_2,a_2,\cdots,s_t,a_t
            '''
            sequenceembedding = torch.concat(
                (statesembedding,actionsembedding),
                dim=-1
            )[:,-1,:]
            positivesamples = actionsembedding[:,-1,:]
            '''
            positive data is [batch_size,embed_dim]
            '''

    
        
        pass