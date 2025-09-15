import torch
import numpy as np

from torch_geometric.data import Data
from torch_geometric.datasets import TUDataset, MoleculeNet

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

# Get TUDataset
def getTU(dataset_name, limit):

    dataset = TUDataset("~/Dataset", name=dataset_name, use_node_attr=True)

    dataset_dg = [d for d in dataset if len(d.x) < limit]

    zero = [d for d in dataset_dg if d.y == 0]
    uno = [d for d in dataset_dg if d.y == 1]

    zero = zero[:min(len(zero), len(uno))]
    uno = uno[:min(len(zero), len(uno))]
    print(f"Zero {len(zero)} | Uno {len(uno)}")
    dataset_dg = zero+uno
    y_g = [0 for _ in range(min(len(zero), len(uno)))]+[1 for _ in range(min(len(zero), len(uno)))]

    X_train, X_test = train_test_split(dataset_dg, test_size=0.25, random_state=0, shuffle=True, stratify=y_g)

    return X_train, X_test


# Get MoleculeNet datasets
def getMol(dataset_name, limit, seed=0):

    dataset = MoleculeNet("~/Dataset", name=dataset_name)

    y_p = np.array([d.y.item() for d in dataset]).reshape(-1, 1)
    y_p = scaler.fit_transform(y_p)
    y_p = torch.from_numpy(y_p).float().reshape(-1, 1)

    data_ = []
    for i in range(len(dataset)):
        try:
            smiles = dataset[i].smiles
            data_.append(Data(x=dataset[i].x, edge_index=dataset[i].edge_index, y=dataset[i].y, smiles=smiles))
        except:
            data_.append(Data(x=dataset[i].x, edge_index=dataset[i].edge_index, y=dataset[i].y))

    dataset_dg = [d for d in data_ if len(d.x) < limit]

    X_train, X_test = train_test_split(dataset_dg, test_size=0.25, random_state=seed)

    return X_train, X_test

