import torch
import random
import numpy as np

from tqdm import tqdm
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.utils import to_scipy_sparse_matrix, get_laplacian

from torch_geometric.utils import degree
from torch_geometric.utils import to_dense_adj


def degreeNode(g):
    d = degree(g.edge_index[0], num_nodes=g.num_nodes).view(-1, 1)
    return d

def get_non_connected_edges(num_nodes, edge_index):
    adj = to_dense_adj(edge_index, max_num_nodes=num_nodes)[0]
    non_connected_edges = (adj == 0).nonzero(as_tuple=False).t()
    non_connected_edges = non_connected_edges[:, non_connected_edges[0] < non_connected_edges[1]]
    return non_connected_edges

def getGraphC(g, g1, ged_2, max_classes, i, j):

    pad_size_g = (max_classes - (g.x.size(0) + g1.x.size(0)))

    mask_sub = torch.zeros((max_classes, max_classes))
    mask_sub[:len(g.x), :len(g1.x)] = 1
    mask_sub = torch.tensor(np.array(mask_sub)).unsqueeze(0)

    mask_add = torch.zeros((max_classes, max_classes))
    mask_add[len(g1.x):len(g1.x) + len(g.x), :len(g.x)] = torch.eye(len(g.x))
    mask_add = torch.tensor(np.array(mask_add)).unsqueeze(0)

    mask_remove = torch.zeros((max_classes, max_classes))
    mask_remove[:len(g1.x), len(g.x):len(g1.x) + len(g.x)] = torch.eye(len(g1.x))
    mask_remove = torch.tensor(np.array(mask_remove)).unsqueeze(0)

    mask_padding = torch.ones((max_classes, max_classes))
    mask_padding[:, :len(g.x)] = 0.0
    mask_padding[:len(g1.x), :] = 0.0
    mask_padding = torch.tensor(np.array(mask_padding)).unsqueeze(0)

    mask_max = torch.ones((max_classes, max_classes)) * 999.0
    mask_max = torch.tensor(np.array(mask_max)).unsqueeze(0)

    general_matrix = torch.cat((mask_sub, mask_add, mask_remove, mask_padding, mask_max)).bool()

    add_empty_nodes = torch.zeros((g1.x.size(0), g.x.size(1)))
    add_empty_nodes1 = torch.zeros((g.x.size(0), g.x.size(1)))

    pad_g = torch.cat((add_empty_nodes, torch.zeros((pad_size_g, g.x.size(1)))))
    pad_g1 = torch.cat((add_empty_nodes1, torch.zeros((pad_size_g, g.x.size(1)))))

    g_tmp = g.clone()
    g_tmp_1 = g1.clone()

    g_tmp.x = torch.cat((g_tmp.x, pad_g))
    g_tmp_1.x = torch.cat((g_tmp_1.x, pad_g1))

    g_tmp.edge_index_inverse = get_non_connected_edges(g_tmp.num_nodes, g_tmp.edge_index)
    g_tmp_1.edge_index_inverse = get_non_connected_edges(g_tmp_1.num_nodes, g_tmp_1.edge_index)

    g_tmp.origin = len(g.x)
    g_tmp_1.origin = len(g1.x)

    num_nodes_ = g_tmp.edge_index.max().item() + 1
    num_nodes_1 = g_tmp_1.edge_index.max().item() + 1

    adj_matrix_ = to_dense_adj(g_tmp.edge_index, max_num_nodes=num_nodes_)[0]
    adj_matrix_1 = to_dense_adj(g_tmp_1.edge_index, max_num_nodes=num_nodes_1)[0]

    adj_padding = torch.zeros((max_classes, max_classes))
    adj_padding_1 = torch.zeros((max_classes, max_classes))

    g_tmp.dg = degreeNode(g_tmp)
    g_tmp_1.dg = degreeNode(g_tmp_1)

    adj_padding[:adj_matrix_.size(0), :adj_matrix_.size(1)] = adj_matrix_
    adj_padding_1[:adj_matrix_1.size(0), :adj_matrix_1.size(1)] = adj_matrix_1

    g_tmp.adj_m = (g_tmp.dg * torch.eye(max_classes) - adj_padding).unsqueeze(0)
    g_tmp_1.adj_m = (g_tmp_1.dg * torch.eye(max_classes) - adj_padding_1).unsqueeze(0)

    g_tmp.i = i
    g_tmp_1.j = j

    return [g_tmp, g_tmp_1, general_matrix, ged_2]


def PairBP(dataset, label, index, max_nodes):
    _test = []

    for i in index:
        for j in index:

            if len(dataset[i].x) >= max_nodes or len(dataset[j].x) >= max_nodes:
                print("Error max nodes")
                exit()

            g_tmp = getGraphC(dataset[i], dataset[j], label[i][j], max_nodes, i, j)
            _test.append(g_tmp)

    return _test

def getMask(GED_train, max_classes):

    _test = []

    for g, g1 in GED_train:

        pad_size = (max_classes - (g.x.size(0) + g1.x.size(0)))
        feature_dim = g1.x.size(1)

        len_g = len(g.x)
        len_g1 = len(g1.x)

        mask_sub = torch.zeros((max_classes, max_classes), dtype=torch.bool)
        mask_sub[:len_g, :len_g1] = 1

        mask_add = torch.zeros((max_classes, max_classes), dtype=torch.bool)
        mask_add[len_g1:len_g1 + len_g, :len_g] = torch.eye(len_g, dtype=torch.bool)

        mask_remove = torch.zeros((max_classes, max_classes), dtype=torch.bool)
        mask_remove[:len_g1, len_g:len_g + len_g1] = torch.eye(len_g1, dtype=torch.bool)

        mask_padding = torch.ones((max_classes, max_classes), dtype=torch.bool)
        mask_padding[:, :len_g] = 0
        mask_padding[:len_g1, :] = 0

        general_matrix = torch.stack([mask_sub, mask_add, mask_remove, mask_padding])


        pad_g = torch.cat((
            torch.zeros((len_g1, feature_dim)),
            -torch.ones((pad_size, feature_dim))
        ))
        pad_g1 = torch.cat((
            torch.zeros((len_g, feature_dim)),
            -torch.ones((pad_size, feature_dim))
        ))

        g_tmp, g_tmp_1 = g.clone(), g1.clone()

        g_tmp.x = torch.cat((g_tmp.x, pad_g))
        g_tmp_1.x = torch.cat((g_tmp_1.x, pad_g1))

        g_tmp.origin = len(g.x)
        g_tmp_1.origin = len(g1.x)

        g_tmp.dg = degreeNode(g_tmp)
        g_tmp_1.dg = degreeNode(g_tmp_1)

        _test.append([g_tmp, g_tmp_1, general_matrix])

    return _test



def getTripletC(dataset_list, limit, max_nodes, g_pivot=None, y_pivot=None):
    if g_pivot is None:
        y_2 = torch.cat([g.y for g in dataset_list]).view(-1)

        indici = torch.arange(len(y_2))

        indici_zero = indici[y_2 == 0]
        indici_ones = indici[y_2 == 1]

        select_zero = np.random.choice(indici_zero, limit)
        select_ones = np.random.choice(indici_ones, limit)

        pivot = np.concatenate((select_zero, select_ones))

        g_pivot = [dataset_list[i] for i in pivot]
        y_pivot = y_2[pivot]

    graph = []

    for i in range(len(dataset_list)):
        g1 = dataset_list[i]

        tmp_pivot = [g1 for _ in range(len(g_pivot))]

        g_tmp = list(zip(tmp_pivot.copy(), g_pivot.copy()))

        m_tmp = getMask(g_tmp, max_nodes)

        graph.append(m_tmp)

    return graph, g_pivot, y_pivot


def getTripletS(dataset_list, limit, max_nodes, g_pivot=None, y_pivot=None, shift=10):
    if g_pivot is None:
        y_2 = torch.cat([g.y for g in dataset_list]).view(-1)
        max_limit = len(y_2) - 1

        idx_2 = torch.argsort(y_2).view(-1)
        idx_1 = [id + random.randint(-shift, shift) for id in range(0, y_2.size(-1), len(idx_2) // limit)]

        idx_1 = np.array(idx_1)
        idx_1[0] = 0
        idx_1[-1] = max_limit

        pivot = idx_2[idx_1]
        g_pivot = [dataset_list[i] for i in pivot]
        y_pivot = y_2[idx_2[idx_1]]

    graph = []

    for i in range(len(dataset_list)):
        g1 = dataset_list[i]

        tmp_pivot = [g1 for _ in range(len(g_pivot))]

        g_tmp = list(zip(tmp_pivot.copy(), g_pivot.copy()))

        m_tmp = getMask(g_tmp, max_nodes)

        graph.append(m_tmp)

    return graph, g_pivot, y_pivot


def getMask(GED_train, max_classes):
    _test = []

    for g, g1 in GED_train:
        pad_size = (max_classes - (g.x.size(0) + g1.x.size(0)))
        feature_dim = g1.x.size(1)

        len_g = len(g.x)
        len_g1 = len(g1.x)

        mask_sub = torch.zeros((max_classes, max_classes), dtype=torch.bool)
        mask_sub[:len_g, :len_g1] = 1

        mask_add = torch.zeros((max_classes, max_classes), dtype=torch.bool)
        mask_add[len_g1:len_g1 + len_g, :len_g] = torch.eye(len_g, dtype=torch.bool)

        mask_remove = torch.zeros((max_classes, max_classes), dtype=torch.bool)
        mask_remove[:len_g1, len_g:len_g + len_g1] = torch.eye(len_g1, dtype=torch.bool)

        mask_padding = torch.ones((max_classes, max_classes), dtype=torch.bool)
        mask_padding[:, :len_g] = 0
        mask_padding[:len_g1, :] = 0

        general_matrix = torch.stack([mask_sub, mask_add, mask_remove, mask_padding])

        pad_g = torch.cat((
            torch.zeros((len_g1, feature_dim)),
            -torch.ones((pad_size, feature_dim))
        ))
        pad_g1 = torch.cat((
            torch.zeros((len_g, feature_dim)),
            -torch.ones((pad_size, feature_dim))
        ))

        g_tmp, g_tmp_1 = g.clone(), g1.clone()

        g_tmp.x = torch.cat((g_tmp.x, pad_g))
        g_tmp_1.x = torch.cat((g_tmp_1.x, pad_g1))

        g_tmp.origin = len(g.x)
        g_tmp_1.origin = len(g1.x)

        g_tmp.dg = degreeNode(g_tmp)
        g_tmp_1.dg = degreeNode(g_tmp_1)

        _test.append([g_tmp, g_tmp_1, general_matrix])

    return _test
