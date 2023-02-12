from torch.utils.data import Dataset
from random import randint

from random import choice,choices
N = 1024


class mydataset(Dataset):
    def __init__(self) -> None:
        super(mydataset,self).__init__()
        self.instalist = list(range(N))
        
    def __len__(self):
        return N
    
    def __getitem__(self,index):
        return self.instalist[index]

if __name__ == "__main__":
    data = mydataset()
    print(choices(data,k=32))
    pass