from dynabench.dataset import DynabenchIterator
from scipy.spatial import KDTree
from sklearn.neighbors import BallTree
from sklearn.metrics import DistanceMetric


import torch
import numpy as np

class DynabenchKNNDataset(torch.utils.data.Dataset):
    def __init__( 
            self,
            split: str="train",
            equation: str="wave",
            structure: str="cloud",
            rollout: int = 16,
            resolution: str="low",
            base_path: str="data",
            k: int = 15):
        
        self.split = split
        self.equation = equation
        self.structure = structure
        self.resolution = resolution
        self.rollout = rollout
        self.base_path = base_path
        self.k = k

        self.iterator = DynabenchIterator(
            base_path=self.base_path,
            split=self.split,
            equation=self.equation,
            structure=self.structure,
            resolution=self.resolution,
            rollout=self.rollout
        )

    
    def __len__(self):
        return len(self.iterator)
    
    def __getitem__(self, index):

        x, y, p = self.iterator[index]

        if self.structure == "grid":
            p = p.reshape(-1, 2)
            x = x.reshape((x.shape[0], x.shape[1], -1))
            x = x.transpose(0, 2, 1)

            y = y.reshape((y.shape[0], y.shape[1], -1))
            y = y.transpose(0, 2, 1)



        p_augmented = np.concatenate([p, 
                                      p + [[0,1]], 
                                      p + [[1,0]],
                                      p + [[0,-1]],
                                      p + [[-1,0]],
                                      p + [[1,-1]],
                                      p + [[-1,1]],
                                      p + [[-1,-1]],
                                      p + [[1,1]]], axis=0)

        
        tree = KDTree(p_augmented)

        distances, n = tree.query(p, k=self.k+1)
        n = n[:, 1:] 
        points_unsqueezed = np.expand_dims(p, axis=1)
        neighbor_points = p_augmented[n]

        n = n % p.shape[0]
        distances = neighbor_points - points_unsqueezed
        

        return x[0].astype(np.float32), y.astype(np.float32), (p.astype(np.float32), n.astype(np.int32), distances.astype(np.float32))
    


if __name__ == "__main__":
    ds = DynabenchKNNDataset(split="train", equation="advection", structure="grid", resolution="high", rollout=4, k=19)

    import tqdm

    x, y, (p, n, d, n_v) = ds[0]
    print(x.shape, y.shape, p.shape, n.shape, d.shape, n_v.shape)
    

    import matplotlib.pyplot as plt

    plt.scatter(p[:, 0], p[:, 1], c="b")
    plt.scatter(p[n[0], 0], p[n[0], 1], c="r")
    plt.show()