import torch
import torch.nn as nn
import scipy
import matplotlib.pyplot as plt

from sklearn.metrics import root_mean_squared_error

device = "cuda"

mseLoss = nn.SmoothL1Loss()

def train_step(model, g1, g2, mask, ged, optim, supervised=True):
    optim.zero_grad()

    g1 = g1.to(device)
    g2 = g2.to(device)
    mask = mask.to(device)
    ged = ged.to(device)

    ged_pred, _ = model.forward(g1, g2, mask, supervised)

    loss_ged = mseLoss(ged_pred.view(-1), ged.view(-1))

    if supervised:
        loss = loss_ged
    else:
        loss = ged_pred.sum()

    loss.backward()
    optim.step()

    return loss, mseLoss(ged_pred.view(-1), ged.view(-1)).mean()


def trainGED(model, d_train, optim, supervised):
    model.train()

    ged_train_sum = 0
    ged_train_loss = 0

    for g1, g2, mask, ged in d_train:

        tps, tpl = train_step(model, g1, g2, mask, ged, optim, supervised)

        ged_train_sum += tps
        ged_train_loss += tpl

    ged_train_sum = ged_train_sum/len(d_train)
    ged_train_loss = ged_train_loss/len(d_train)

    return ged_train_sum, ged_train_loss



def getPerformance(model, d_test, supervised):
    real_ged_l = []
    pred_ged_l = []

    g1_list = []
    g2_list = []

    model.eval()

    for g1, g2, mask, ged in d_test:
        with torch.no_grad():
            g1 = g1.to(device)
            g2 = g2.to(device)
            mask = mask.to(device)
            ged = ged.to(device)

            ged_pred, _ = model.forward(g1, g2, mask, supervised)

            real_ged_l.append(ged)
            pred_ged_l.append(ged_pred)

            g1_list.extend(g1.i)
            g2_list.extend(g2.j)


    real_ged_l = torch.cat(real_ged_l).int()
    pred_ged_l = torch.cat(pred_ged_l).squeeze().int()

    real_ged_l = real_ged_l.cpu().detach().numpy()
    pred_ged_l = pred_ged_l.cpu().detach().numpy()

    mse_ged = root_mean_squared_error(real_ged_l, pred_ged_l)

    corr, cp = scipy.stats.kendalltau(real_ged_l, pred_ged_l)
    r2, _ = scipy.stats.pearsonr(real_ged_l, pred_ged_l)

    return mse_ged, corr, r2


def getPerformanceTest(model, d_test, supervised, plot=False, dataset="", conf=0, part=0):

    real_ged_l = []
    pred_ged_l = []

    g1_list = []
    g2_list = []

    model.eval()

    for g1, g2, mask, ged in d_test:
        with torch.no_grad():
            g1 = g1.to(device)
            g2 = g2.to(device)
            mask = mask.to(device)
            ged = ged.to(device)

            ged_pred, _ = model.forward(g1, g2, mask, supervised)

            real_ged_l.append(ged)
            pred_ged_l.append(ged_pred)

            g1_list.extend(g1.i)
            g2_list.extend(g2.j)

    real_ged_l = torch.cat(real_ged_l).int()
    pred_ged_l = torch.cat(pred_ged_l).squeeze().int()

    real_ged_l = real_ged_l.cpu().detach().numpy()
    pred_ged_l = pred_ged_l.cpu().detach().numpy()

    mse = root_mean_squared_error(real_ged_l, pred_ged_l)

    if plot:
        plt.scatter(real_ged_l, pred_ged_l)
        plt.xlabel("Real GED")
        plt.ylabel("GED predicted by U-GEDAN")
        plt.savefig(f"results/{dataset}/config_{conf}_fold_{part}.png")
        plt.close()

    tau, cp = scipy.stats.kendalltau(real_ged_l, pred_ged_l)
    r2, _ = scipy.stats.pearsonr(real_ged_l, pred_ged_l)

    return mse, tau, r2