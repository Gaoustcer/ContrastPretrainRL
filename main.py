from utils.trajdataset import Trajdataset

if __name__ == "__main__":
    dataset = Trajdataset(device="cuda")
    for traj in dataset:
        print(traj.observations.shape,traj.actions.shape)