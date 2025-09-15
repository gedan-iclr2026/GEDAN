import sys
import time
import torch
import random
import argparse
import numpy as np

from torch_geometric.loader import DataLoader

from src.utils.dataset_load import getMol
from src.utils.data_creation import getMask
from src.utils.mol_analysis import getValues, getSingleVal, plotGraph

def init(device_):

    seed = 0

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    dataset_name = "FreeSolv"
    size_graph = 32

    if dataset_name == "FreeSolv":
        GED_train, GED_test = getMol("FreeSolv", size_graph, seed)

    max_nodes = 64

    model = torch.load(f"src/checkpoints/model_molecules.pt", weights_only=False, map_location=torch.device(device_))

    return model, GED_train, GED_test, max_nodes, device_


def single(model, GED_train, mol_s, mol_t, max_nodes, device):

    start = time.time()

    y_val = np.array([g.y.item() for g in GED_train])
    idx_sort = np.argsort(y_val)

    g1 = GED_train[idx_sort[mol_s]]
    g2 = GED_train[idx_sort[mol_t]]

    print(f"{g1.smiles} ({g1.y.item():.3f}) --> {g2.smiles}")

    costo_ = getSingleVal(model, g1, g2, len(g1.x), max_nodes, device)

    print(f"Analysis completed in {time.time()-start:.2f} seconds on CPU")

    plotGraph(g1, costo_, g2.smiles)


def multi(model, GED_train, mol_s, max_nodes, device):

    start = time.time()

    y_val = np.array([g.y.item() for g in GED_train])
    idx_sort = np.argsort(y_val)

    costo_ = 0
    g1 = GED_train[idx_sort[mol_s]]

    print(f"{g1.smiles} ({g1.y.item():.3f}) --> all")

    data_ = []
    for ival in range(len(GED_train)):
        data_.extend(getMask([[g1, GED_train[idx_sort[ival]]]], max_nodes))

    loader_ = DataLoader(data_, batch_size=1024, shuffle=False)

    model.eval()

    with torch.no_grad():
        for a1, a2, a_mask in loader_:
            _, _, perm, levels, x_v, _ = model.getMatchAll(a1.to(device), a2.to(device), a_mask.to(device))

            levels = levels.view(-1, len(a1), levels.size(1), levels.size(2))
            levels = levels.swapaxes(0, 1)

            x_v = x_v.view(x_v.size(0), len(a1), -1, x_v.size(-1))

            a1 = a1.cpu().detach()
            a_mask = a_mask.cpu().detach()
            x_v = x_v.cpu().detach()
            levels = levels.cpu().detach()
            perm = np.round(perm.cpu().detach().numpy())

            for grafi in range(len(a1)):
                costo_ += getValues(a1[grafi], a_mask[grafi], levels[grafi], perm[grafi],
                                    x_v[:, [grafi]].squeeze(1), len(g1.x))

    print(f"Analysis completed in {time.time()-start:.2f} seconds on {device}")

    plotGraph(g1, costo_, "all")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Molecule Analysis Script")

    parser.add_argument("--input", type=str, help="Input molecule index")
    parser.add_argument("--target", type=str, help="Target molecule index or \"all\"")

    args = parser.parse_args()

    mol = int(args.input)
    type_proc = args.target

    model, GED_train, GED_test, max_nodes, device = init("cpu")

    if type_proc == "all":
        multi(model, GED_train, mol, max_nodes, device)
    else:
        single(model, GED_train, mol, int(type_proc), max_nodes, device)






