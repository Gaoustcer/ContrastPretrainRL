import torch.nn as nn
import torch

class infonce(nn.Module):
    def __init__(self,querydim = 32,keydim = 16) -> None:
        super(infonce,self).__init__()
        self.querytokeytransformer = nn.Sequential(
            nn.Linear(querydim,16),
            nn.ReLU(),
            nn.Linear(16,keydim)
        )
    
    def forward(self,
                query:torch.Tensor,
                positivekeys:torch.Tensor,
                negativekeys:torch.Tensor,
                tau = 0.17):
        '''
        query [N,querydim]
        positivekeys [N,querydim]
        negativekeys [M,querydim]
        '''
        if query.dim() == 1:
            query = query.unsqueeze(0)
        if positive.dim() == 1:
            positive = positive.unsqueeze(0)
        query = self.querytokeytransformer(query)
        assert query.shape == positivekeys.shape
        # query = self.querytokeytransformer(query)
        similaritytrans = torch.mm(query,positivekeys.T)
        similaritypositivepairs = similaritytrans.diag().unsqueeze(-1)
        '''
        similaritypostivepairs [N]
        '''
        query_negativesimilarity = torch.mm(query,negativekeys.T)
        '''
        query_negativesimilarity [N,M]
        '''
        similaritymatrix = torch.concat(
            (similaritypositivepairs,query_negativesimilarity),
            dim = -1
        )/tau
        similaritymatrix = torch.exp(similaritymatrix)
        simloss = -torch.log(
            similaritymatrix[:,0]/torch.sum(similaritymatrix,dim=-1)
        )
        return simloss


if __name__ == "__main__":
    query = torch.rand(17,16)
    positive = torch.rand(17,32)
    negative = torch.rand(22,32)
    nceloss = infonce(querydim=16,keydim=32)
    loss = nceloss(query,positive,negative)
    print(loss.shape)
