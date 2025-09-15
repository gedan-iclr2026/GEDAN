import sys
import json
import torch
import random
import argparse
import numpy as np

from sklearn.model_selection import KFold
from torch_geometric.loader import DataLoader

from src.utils.data_creation import PairBP
from src.models.model_train_GED import getPerformanceTest

torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

random.seed(42)

np.random.seed(42)


def run(dataset_n_, supervisioned_, device_):

    device = device_

    supervisioned = supervisioned_
    s_text = "S"
    if not supervisioned:
        s_text = "U"

    dataset_n = dataset_n_

    with open(f'src/config/configurations_{s_text}.json', 'r') as f:
        conf_costs = json.load(f)

    with open(f'src/config/{dataset_n}_{s_text}.json', 'r') as f:
        conf_dataset = json.load(f)

    result_m = []
    result_t = []
    result_p = []

    max_nodes = conf_costs["parameters"]["max_nodes"]
    config_model = conf_dataset["gedan"]
    print(config_model, "\n")

    for conf in range(0, 5):

        diz_w = conf_costs["costs"][str(conf)]
        print(f"Edit Cost Configuration: {diz_w}")

        data_graph = torch.load(f"data/{dataset_n}/{dataset_n}_data.pt")
        ged_graph = np.load(f"data/{dataset_n}/{dataset_n}_GED_conf_{conf}.npy")

        partizione = 0

        test_mse = []
        test_tau = []
        test_per = []

        for train_index, test_index in KFold(n_splits=5, random_state = 0, shuffle=True).split(data_graph):

            graphs_test = PairBP(data_graph, ged_graph, test_index, max_nodes)

            if supervisioned:
                f_load = f"src/checkpoints/{dataset_n}/{conf}_checkpoint_{partizione}_SM.pt"
            else:
                f_load = f"src/checkpoints/{dataset_n}/{conf}_checkpoint_{partizione}_UM.pt"

            model_GED = torch.load(f_load, weights_only=False).to(device)
            model_GED.eval()

            d_test = DataLoader(graphs_test, batch_size=config_model["bs_test"], shuffle=False)

            mse_ged, corr, r2 = getPerformanceTest(model_GED, d_test, supervisioned, plot=True, dataset=dataset_n, conf=conf, part=partizione)

            print(f"Conf: {conf} | Fold: {partizione} | RMSE: {mse_ged:.3f}\t T: {corr:.3f}\t P: {r2:.3f}")

            test_mse.append(mse_ged)
            test_tau.append(corr)
            test_per.append(r2)

            partizione += 1

        test_mse = np.array(test_mse)
        test_tau = np.array(test_tau)
        test_per = np.array(test_per)

        result_m.append(test_mse)
        result_t.append(test_tau)
        result_p.append(test_per)

        print(f"Avg: RMSE {test_mse.mean():.2f}\t T {test_tau.mean():.3f}\t P {test_per.mean():.3f}")
        print(f"Std: RMSE {test_mse.std():.3f}\t T {test_tau.std():.3f}\t P {test_per.std():.3f}")
        print()

    result_m = np.array(result_m)
    result_t = np.array(result_t)
    result_p = np.array(result_p)

    print("Total")
    print(f"RMSE {result_m.mean():.2f}\t T {result_t.mean():.3f}\t P {result_p.mean():.3f}")



if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Script to test GED results")

    parser.add_argument("--dataset", type=str, help="Dataset [AIDS, MUTAG, PTC_MR]")
    parser.add_argument("--model", type=str, help="Type of GEDAN [S, U]")

    args = parser.parse_args()

    dataset_name = args.dataset
    type_GEDAN = args.model
    flag_GEDAN = False

    if dataset_name not in ["MUTAG", "AIDS", "PTC_MR"]:
        print("Dataset not included")
        exit()

    if type_GEDAN not in ["S", "U"]:
        print("GEDAN error, choose S or U")
        exit()
    if type_GEDAN == "S":
        flag_GEDAN = True

    run(dataset_name, flag_GEDAN, "cuda")
