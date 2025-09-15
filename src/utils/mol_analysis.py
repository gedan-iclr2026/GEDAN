import torch
import matplotlib
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from src.utils.data_creation import getMask
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader


def getList(types, costs):

    list_ = []

    unique = torch.unique(types, dim=0)
    values = np.zeros(len(unique))

    for t in range(len(types)):

        type_ = types[t]
        val = costs[t]

        for i in range(len(unique)):
            l = unique[i]
            if torch.equal(type_, l):
                values[i] = max(values[i], val)
                continue

    for t in types:
        for i in range(len(unique)):
            if torch.equal(t, unique[i]):
                list_.append(values[i])

    return np.array(list_)


def getSingleVal(model, g1, g2, nodes, max_nodes, device):

    data_ = getMask([[g1, g2]], max_nodes)

    model.eval()
    loader_ = DataLoader(data_, batch_size=1, shuffle=False)

    with torch.no_grad():
        for a1, a2, a_mask in loader_:
            ged, _, perm, levels, x_v, _ = model.getMatchAll(a1.to(device), a2.to(device), a_mask.to(device))
            levels = levels.cpu().detach()
            perm = np.round(perm[0].cpu().detach().numpy())

            mask = a_mask[0]
            mask = mask[0]+mask[1]+mask[2]+mask[3]
            mask = ~mask
            lt = (levels[0]-(mask*levels[0].max()))* perm
            lt[lt<1e-5] = 0

            costo_t = lt.sum(-1)[:nodes]
            costo = getList(a1.x[:nodes], costo_t)

            for li in range(1, 4):

                mask = a_mask[0]
                mask = mask[0] + mask[1] + mask[2] + mask[3]
                mask = ~mask
                lt = (levels[li] - (mask * levels[li].max())) * perm
                lt[lt < 1e-5] = 0

                costo_t = lt.sum(-1)[:nodes]
                costo = getList(x_v[li-1][:nodes], costo_t)

            if np.isnan(costo.sum()):
                costo = np.zeros_like(costo)

    return costo


def plotGraph(g1, costo_, type):

    costo_ = (costo_-costo_.min()) / (costo_.max()-costo_.min())

    costo_[costo_>0.45] = 1
    costo_[(costo_ >= 0.25) & (costo_ <= 0.45)] = 0.5
    costo_[(costo_ >= 0.05) & (costo_ < 0.25)] = 0.3
    costo_[costo_<0.05] = 0

    G = to_networkx(g1, node_attrs=["x"])

    pos_g1 = nx.spring_layout(G, seed=0)

    plt.figure(figsize=(6, 6))
    cmap = matplotlib.colormaps["inferno"]
    norm = mcolors.Normalize(vmin=costo_.min(), vmax=costo_.max())
    g_colors = [cmap(norm(val)) for val in costo_]
    nx.draw_networkx(G, pos=pos_g1, with_labels=False, node_color=g_colors)
    plt.title(g1.smiles)
    plt.tight_layout()
    plt.savefig(f"results/Graphs/{g1.smiles}_GEDAN_{type}.png")



def getValues(a1, a_mask, levels, perm, x_v, nodes):

    mask = a_mask
    mask = mask[0]+mask[1]+mask[2]+mask[3]
    mask = ~mask
    lt = (levels[0] - (mask * levels[0].max())) * perm
    lt[lt<1e-5] = 0

    costo_t = lt.sum(-1)[:nodes]
    costo = getList(a1.x[:nodes], costo_t)

    for li in range(1, 4):

        mask = a_mask
        mask = mask[0] + mask[1] + mask[2] + mask[3]
        mask = ~mask
        lt = (levels[li] - (mask * levels[li].max())) * perm
        lt[lt < 1e-5] = 0

        costo_t = lt.sum(-1)[:nodes]
        costo = getList(x_v[li-1][:nodes], costo_t)


    if np.isnan(costo.sum()):
        costo = np.zeros_like(costo)

    return costo
