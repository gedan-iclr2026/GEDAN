import json
import torch
import numpy as np

from torch import optim
from torch_geometric.loader import DataLoader
from tqdm import tqdm

from src.models.model_train_GED import getPerformance, trainGED
from src.utils.data_creation import PairBP
from sklearn.model_selection import KFold, train_test_split

from src.models.GEDAN import GEDAN
from src.models.Sinkhorn.sinkhorn_model import GumbelSinkhornTrainableModel

device = "cuda"

def train_GED(dataset_n, supervised, idx_cost):

    print("GED", dataset_n, supervised)

    s_text = "S"
    model_text = "S-GEDAN"
    if not supervised:
        s_text = "U"
        model_text = "U-GEDAN"

    with open(f'src/config/configurations_{s_text}.json', 'r') as f:
        conf_costs = json.load(f)

    with open(f'src/config/{dataset_n}_{s_text}.json', 'r') as f:
        conf_dataset = json.load(f)

    diz_w = conf_costs["costs"][str(idx_cost)]
    max_nodes = conf_costs["parameters"]["max_nodes"]
    max_patience = conf_costs["parameters"]["max_patience"]
    max_epoch = conf_costs["parameters"]["max_epoch"]
    config_gumbel = conf_dataset["gumbel"]
    config_model = conf_dataset["gedan"]
    print()
    print("Costs\t\tNode\tEdge")
    print(f"Substitution\t{diz_w[0]}\tNA")
    print(f"Insertion\t{diz_w[1]}\t{diz_w[2]}")
    print(f"Deletion\t{diz_w[3]}\t{diz_w[4]}")
    print()
    print("Model config:")
    print(config_model)
    print()

    data_graph = torch.load(f"data/{dataset_n}/{dataset_n}_data.pt")
    ged_graph = np.load(f"data/{dataset_n}/{dataset_n}_GED_conf_{idx_cost}.npy")

    fold = 0

    plt_test_mse = []
    plt_test_tau = []
    plt_test_per = []

    print(f"---------------------| Cost configuration: {idx_cost} |---------------------\n")

    for train_index, test_index in KFold(n_splits=5, shuffle=True).split(data_graph):

        best_GED_ = 99999
        best_res = ""

        train_index, val_index = train_test_split(train_index, test_size=0.1)

        graphs_test = PairBP(data_graph, ged_graph, test_index, max_nodes)
        graphs_train = PairBP(data_graph, ged_graph, train_index, max_nodes)
        graphs_val = PairBP(data_graph, ged_graph, val_index, max_nodes)

        patience = 0

        gumbel = GumbelSinkhornTrainableModel(num_iters= config_gumbel["num_iters"],
                                              init_temperature= config_gumbel["num_iters"],
                                              final_temperature= config_gumbel["num_iters"],
                                              noise_factor=config_gumbel["noise_factor"],
                                              num_epochs=max_epoch,
                                              input_dim=max_nodes).to(device)

        gumbel.load_state_dict(torch.load(f"src/models/Sinkhorn/model_32O.pt",
                                          map_location=torch.device(device),
                                          weights_only=False))

        model_GED = GEDAN(  config_model["input_size"],
                            config_model["latent_space"],
                            config_model["n_layer"],
                            max_nodes,
                            gumbel,
                            supervised,
                            diz_w=diz_w,
                            scale_p=1).to(device)
        if fold == 0:
            num_trainable_params = sum(p.numel() for p in model_GED.parameters() if p.requires_grad)
            print(f"Number of parameters ({model_text}): {num_trainable_params}\n")

        optimizer = optim.Adam([
            {'params': [p for name, p in model_GED.named_parameters() if "sinkhorn" not in name],
             'lr': config_model["lr"]}
        ])

        d_train = DataLoader(graphs_train, batch_size=128, shuffle=True)
        d_val = DataLoader(graphs_val, batch_size=32, shuffle=False)
        d_test = DataLoader(graphs_test, batch_size=32, shuffle=False)

        for _ in tqdm(range(max_epoch), desc="Epoch"):
            _, _ = trainGED(model_GED, d_train, optimizer, supervised)
            mse_ged_val, corr, r2 = getPerformance(model_GED, d_val, supervised)


            if mse_ged_val < best_GED_:
                best_GED_ = mse_ged_val
                patience = 0
                best_mse, best_corr, best_r2 = getPerformance(model_GED, d_test, supervised)
                best_res = f"Fold {fold}: RMSE:{best_mse:.3f},\t T:{best_corr:.3f},\t P:{best_r2:.3f}"

                if supervised:
                    torch.save(model_GED.state_dict(), f"results/{dataset_n}/ck/{idx_cost}_checkpoint_{fold}_S.pt")
                else:
                    torch.save(model_GED.state_dict(), f"results/{dataset_n}/ck/{idx_cost}_checkpoint_{fold}_U.pt")

            else:
                patience += 1
                if patience > max_patience:
                    break

        plt_test_mse.append(best_mse)
        plt_test_tau.append(best_corr)
        plt_test_per.append(best_r2)

        print(best_res)
        print(flush=True)
        fold += 1

    plt_test_mse = np.array(plt_test_mse)
    plt_test_tau = np.array(plt_test_tau)
    plt_test_per = np.array(plt_test_per)

    print(idx_cost)
    print(f"MSE {plt_test_mse.mean():.3f}, Tau {plt_test_tau.mean():.3f}, Pear {plt_test_per.mean():.3f}")
    print(f"MSE {plt_test_mse.std():.3f}, Tau {plt_test_tau.std():.3f}, Pear {plt_test_per.std():.3f}")

    print()