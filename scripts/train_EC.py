import json
import math
import random
import warnings
warnings.filterwarnings("ignore")

import torch
import torch.optim as optim
from torch_geometric.loader import DataLoader
from sklearn.model_selection import train_test_split
from src.models.GEDAN import GEDAN, ClassModelRegression
from src.models.Sinkhorn.sinkhorn_model import GumbelSinkhornTrainableModel
from src.models.model_train_EC import trainRegression, evalRegression, evalBinary, getPerformanceG, trainBinary
from src.utils.data_creation import getTripletS, getTripletC
from src.utils.dataset_load import getMol


device = "cuda"

def train_EC(dataset_name):

    with open(f'src/config/{dataset_name}.json', 'r') as f:
        conf_costs = json.load(f)

    gumbel = conf_costs["gumbel"]
    gedan = conf_costs["gedan"]
    options = conf_costs["options"]
    batch_size = options["batch_size"]

    print("\nGEDAN")
    print(gedan)
    print("\nGumbel-Sinkhorn")
    print(gumbel)
    print("\nHyper parameters")
    print(options)

    limit_select = options["limit_select"]
    regression = options["regression"]
    output_y = options["y_output"]
    epochs = options["epochs"]
    refresh = options["refresh"]

    size_graph = 32
    max_nodes = size_graph*2

    if dataset_name == "FreeSolv":
        GED_train, GED_test = getMol("FreeSolv", size_graph, 0)

    elif dataset_name == "BBBP":
        GED_train, GED_test = getMol("BBBP", size_graph, 0)
        regression = False
    else:
        print("\nDataset not included, please add it manually")
        exit()


    print("\nTrain graphs", len(GED_train), "\tTest graphs", len(GED_test))
    random.shuffle(GED_train)
    GED_train, GED_val = train_test_split(GED_train, test_size=0.25)
    random.shuffle(GED_test)

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
                  diz_w=list(torch.rand(5).numpy()+1)).to(device)

    if regression:
        p_number = len(GED_train) / (len(GED_train) // limit_select)
        p_number = math.ceil(p_number)
        classificator = ClassModelRegression(p_number, output_y).to(device)
    else:
        classificator = ClassModelRegression(limit_select*2, output_y, classification = True).to(device)
    
    
    o1 = optim.Adam(list(model.parameters()) + list(classificator.parameters()), lr=0.01)

    if regression:
        graphs_train, g_pivot, y_pivot = getTripletS(GED_train, limit_select, max_nodes)
        graphs_test, g_pivot, y_pivot = getTripletS(GED_test, limit_select, max_nodes, g_pivot, y_pivot)
    else:
        graphs_train, g_pivot, y_pivot = getTripletC(GED_train, limit_select, max_nodes)
        graphs_test, g_pivot, y_pivot = getTripletC(GED_test, limit_select, max_nodes, g_pivot, y_pivot)

    train_loader = DataLoader(graphs_train, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(graphs_test, batch_size=256)

    for epoch in range(epochs):

        if epoch%refresh == 0:
            if regression:
                graphs_train, g_pivot, y_pivot = getTripletS(GED_train, limit_select, max_nodes)
                graphs_test, g_pivot, y_pivot = getTripletS(GED_test, limit_select, max_nodes, g_pivot, y_pivot)
            else:
                graphs_train, g_pivot, y_pivot = getTripletC(GED_train, limit_select, max_nodes)
                graphs_test, g_pivot, y_pivot = getTripletC(GED_test, limit_select, max_nodes, g_pivot, y_pivot)

            train_loader = DataLoader(graphs_train, batch_size=batch_size, shuffle=True)
            test_loader = DataLoader(graphs_test, batch_size=256)

        if regression:
            loss_ged_t, loss_y_t, real_y, pred_y = trainRegression(model, classificator, train_loader, o1, temperatura=0.5)
        else:
            loss_ged_t, loss_y_t, real_y, pred_y = trainBinary(model, classificator, train_loader, o1)


        if regression:
            loss_ged_v, loss_y_v, real_y_v, pred_y_v = evalRegression(model, classificator, test_loader, temperatura=0.5)

            getPerformanceG(epoch, loss_ged_t, loss_ged_v,
                            real_y, pred_y, real_y_v, pred_y_v, loss_y_t=loss_y_t, loss_y_v=loss_y_v)
        else:
            loss_ged_v, loss_y_v, real_y_v, pred_y_v = evalBinary(model, classificator, test_loader)

            getPerformanceG(epoch, loss_ged_t, loss_ged_v,
                            real_y, pred_y, real_y_v, pred_y_v,
                            loss_y_t=loss_y_t, loss_y_v=loss_y_v, regression = False)

        torch.save(model.state_dict(), f"results/{dataset_name}/ck/ckp_{epoch}.pt")