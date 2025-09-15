import sys
import json
import random
import torch
import argparse
import numpy as np

from tqdm import tqdm
from torch_geometric.loader import DataLoader

from src.models.GEDAN import GEDAN
from src.models.Sinkhorn.sinkhorn_model import GumbelSinkhornTrainableModel

from src.utils.ML_embedding import getPredictionEmbedding
from src.utils.dataset_load import getMol
from src.utils.data_creation import getTripletS, getTripletC, getMask

device = "cuda"

seed = 0
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


def getCouplePair(d1, d2, batch_size, model, S_GEDAN, sez, max_nodes):

    ged_tmp = []
    label_ = []

    for g1 in tqdm(d1):
        data_ = []
        graph_ = []

        g_tmp = [g1 for _ in d2]

        elem = list(zip(g_tmp, d2))
        elem_mask = getMask(elem, max_nodes)

        data_.extend(elem_mask)
        label_.append(g1.y.item())

        loader_ = DataLoader(data_, batch_size=batch_size, shuffle=False)

        model.eval()

        with torch.no_grad():
            for a1, a2, a_mask in loader_:

                if S_GEDAN:
                    ged_, _, = model.supervisionedGED(a1.to(device), a2.to(device), a_mask.to(device))
                else:
                    ged_, _, = model.unsupervisionedGED(a1.to(device), a2.to(device), a_mask.to(device))

                graph_.extend(ged_.cpu().detach().numpy())

        ged_tmp.append(graph_)

    ged_tmp = np.array(ged_tmp).squeeze()

    if S_GEDAN:
        np.save(f"results/{dataset_name}/embedding/GEDAN/dati_SGEDAN_{sez}.npy", np.array(ged_tmp))
        np.save(f"results/{dataset_name}/embedding/GEDAN/label_SGEDAN_{sez}.npy", np.array(label_))
    else:
        np.save(f"results/{dataset_name}/embedding/GEDAN/dati_UGEDAN_{sez}.npy", np.array(ged_tmp))
        np.save(f"results/{dataset_name}/embedding/GEDAN/label_UGEDAN_{sez}.npy", np.array(label_))


    return np.array(ged_tmp), np.array(label_)



def getCouplePair2(d1, d2, batch_size, model, S_GEDAN, sez, max_nodes):

    ged_tmp = []
    label_ = []

    data_ = []

    for g1 in d1:
        g_tmp = [g1 for _ in d2]

        elem = list(zip(g_tmp, d2))
        elem_mask = getMask(elem, max_nodes)

        data_.extend(elem_mask)
        label_.append(g1.y.item())

    loader_ = DataLoader(data_, batch_size=batch_size, shuffle=False)

    model.eval()

    with torch.no_grad():
        graph_ = []
        for a1, a2, a_mask in tqdm(loader_):

            if S_GEDAN:
                ged_, _, = model.supervisionedGED(a1.to(device), a2.to(device), a_mask.to(device))
            else:
                ged_, _, = model.unsupervisionedGED(a1.to(device), a2.to(device), a_mask.to(device))

            graph_.extend(ged_.cpu().detach().numpy())

        ged_tmp.append(graph_)

    ged_tmp = np.array(ged_tmp).squeeze().reshape(len(d1), len(d2))

    if S_GEDAN:
        np.save(f"results/{dataset_name}/embedding/GEDAN/dati_SGEDAN_{sez}.npy", np.array(ged_tmp))
        np.save(f"results/{dataset_name}/embedding/GEDAN/label_SGEDAN_{sez}.npy", np.array(label_))
    else:
        np.save(f"results/{dataset_name}/embedding/GEDAN/dati_UGEDAN_{sez}.npy", np.array(ged_tmp))
        np.save(f"results/{dataset_name}/embedding/GEDAN/label_UGEDAN_{sez}.npy", np.array(label_))


    return np.array(ged_tmp), np.array(label_)


def getPivot(dataset_name):

    with open(f'src/config/{dataset_name}.json', 'r') as f:
        conf_costs = json.load(f)

    options = conf_costs["options"]

    max_nodes = 64

    limit_select = options["limit_select"]-1
    regression = options["regression"]

    if regression == 1:
        regression = True
    else:
        regression = False

    if dataset_name == "FreeSolv":
        GED_train, GED_test = getMol("FreeSolv", 32, seed)

    if dataset_name == "BBBP":
        GED_train, GED_test = getMol("BBBP", 32, seed)
        regression = False

    random.shuffle(GED_train)
    random.shuffle(GED_test)

    pivots = []
    pivots_y = []
    for _ in range(3):
        if regression:
            _, pivot, y_pivot = getTripletS(GED_train, limit_select, max_nodes)
        else:
            _, pivot, y_pivot = getTripletC(GED_train, limit_select, max_nodes)

        pivots.append(pivot)
        pivots_y.append(y_pivot)

    return GED_train, GED_test, pivots

def run(dataset_name, S_GEDAN, GED_train, GED_test, pivots, device):

    with open(f'src/config/{dataset_name}.json', 'r') as f:
        conf_costs = json.load(f)

    gumbel = conf_costs["gumbel"]
    gedan = conf_costs["gedan"]
    options = conf_costs["options"]
    batch_size = options["batch_size"]

    diz_w = [1.0, 1.0, 1.0, 1.0, 1.0]


    gumbel = GumbelSinkhornTrainableModel(num_iters=gumbel["num_iters"],
                                          num_epochs=gumbel["epoch"],
                                          init_temperature=gumbel["init_temperature"],
                                          final_temperature=gumbel["final_temperature"],
                                          input_dim=64).to(device)

    gumbel.load_state_dict(torch.load(f"src/models/Sinkhorn/model_64O.pt",
                                       map_location=torch.device(device),
                                       weights_only=False))

    model = GEDAN(gedan["input_size"],
                  gedan["latent_space"],
                  gedan["n_layer"],
                  64,
                  gumbel,
                  True,
                  diz_w=diz_w).to(device)


    if S_GEDAN:
        print("S-GEDAN")

        if dataset_name == "BBBP":
            model.load_state_dict(torch.load(f"src/checkpoints/ckp_BBBP.pt"), strict=False)

        if dataset_name == "FreeSolv":
            model.load_state_dict(torch.load(f"src/checkpoints/ckp_FreeSolv.pt"), strict=False)



    else:
        print("U_GEDAN")

    model.eval()

    for p in range(len(pivots)):
        print(f"Pivot: {p+1}")
        pivot = pivots[p]
        print(f"Train vs Train [{len(GED_train)}x{len(pivot)}]")
        _, _ = getCouplePair(GED_train, pivot, batch_size, model, S_GEDAN, f"train_train_{p}", 64)
        print(f"Test vs Train [{len(GED_test)}x{len(pivot)}]")
        _, _ = getCouplePair(GED_test, pivot, batch_size, model, S_GEDAN, f"test_train_{p}", 64)
        print()


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Edit Cost Optimization Test Script")

    parser.add_argument("--dataset", type=str, help="Dataset [BBBP, FreeSolv]")
    parser.add_argument("--inference", type=bool, default=False, help="Generate embeddings")

    args = parser.parse_args()

    dataset_name = args.dataset

    if dataset_name not in ["FreeSolv", "BBBP"]:
        print("Dataset not included")
        exit()

    GED_train, GED_test, pivots = getPivot(dataset_name)

    if args.inference:
        run(dataset_name, True, GED_train, GED_test, pivots, "cuda")
        print()
        run(dataset_name, False, GED_train, GED_test, pivots, "cuda")

    getPredictionEmbedding(dataset_name)