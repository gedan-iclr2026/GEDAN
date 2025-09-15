import torch
import torch.nn as nn
import numpy as np
import scipy

import torch.nn.functional as F

from tqdm import tqdm
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score, roc_auc_score

device = "cuda"

mseLoss = nn.MSELoss(reduce=False)
mseLossr = nn.MSELoss()
#bceLoss = nn.BCEWithLogitsLoss()
bceLoss = nn.CrossEntropyLoss()

def trainRegression(model, classificator, d_train, o1, temperatura=1):

    loss_ged_t = 0
    loss_y_t = 0

    real_p = []
    pred_p = []

    for dati in tqdm(d_train):

        o1.zero_grad()

        ged_t = []
        ged_r = []
        y_r = []
        y_p = []

        for pivot in dati:
            g1, g2, mask = pivot

            ged_p, _ = model(g1.to(device), g2.to(device), mask.to(device), True, smooth=False)

            ged_t.append(ged_p.unsqueeze(-1))
            ged_r.append(abs(g1.y - g2.y))
            y_r.append(g1.y)
            y_p.append(g2.y)

        ged_t = torch.cat(ged_t, -1)
        ged_r = torch.cat(ged_r, -1)

        pos_indices = torch.argmin(ged_r, -1).unsqueeze(1)

        log_probs = F.log_softmax(-ged_t / temperatura, dim=1)
        kl_loss = -log_probs.gather(1, pos_indices).squeeze(1)

        kl_loss = kl_loss.mean()

        y_r = torch.cat(y_r, -1).mean(-1)

        y_pred = classificator(ged_t).view(-1)

        loss_r = mseLossr(y_pred, y_r)

        real_p.append(y_r)
        pred_p.append(y_pred)

        loss = kl_loss + loss_r

        loss_ged_t += kl_loss
        loss_y_t += loss_r

        loss.backward()
        o1.step()

    loss_ged_t = loss_ged_t / len(d_train)
    loss_y_t = loss_y_t / len(d_train)

    real_p = torch.cat(real_p)
    pred_p = torch.cat(pred_p)

    return loss_ged_t, loss_y_t, real_p, pred_p


def evalRegression(model, classificator, d_train, temperatura=1):
    loss_ged_t = 0
    loss_y_t = 0

    real_p = []
    pred_p = []

    with torch.no_grad():
        for dati in d_train:

            ged_t = []
            ged_r = []
            y_r = []
            y_p = []

            for pivot in dati:
                g1, g2, mask = pivot

                ged_p, _ = model(g1.to(device), g2.to(device), mask.to(device), True, smooth=False)

                ged_t.append(ged_p.unsqueeze(-1))
                ged_r.append(abs(g1.y - g2.y))
                y_r.append(g1.y)
                y_p.append(g2.y)

            ged_t = torch.cat(ged_t, -1)
            ged_r = torch.cat(ged_r, -1)
            y_p = torch.cat(y_p, -1)
            y_r = torch.cat(y_r, -1).mean(-1)

            pos_indices = torch.argmin(ged_r, -1).unsqueeze(1)

            log_probs = F.log_softmax(-ged_t / temperatura, dim=1)
            kl_loss = -log_probs.gather(1, pos_indices).squeeze(1)
            kl_loss = kl_loss.mean()

            y_pred = classificator(ged_t).view(-1)

            loss_r = mseLossr(y_pred, y_r)

            real_p.append(y_r)
            pred_p.append(y_pred)

            loss_ged_t += kl_loss
            loss_y_t += loss_r

    loss_ged_t = loss_ged_t / len(d_train)
    loss_y_t = loss_y_t / len(d_train)

    real_p = torch.cat(real_p)
    pred_p = torch.cat(pred_p)

    return loss_ged_t, loss_y_t, real_p, pred_p

def trainBinary(model, classificator, d_train, o1, temperatura=1):
    loss_ged_t = 0
    loss_y_t = 0

    real_p = []
    pred_p = []

    for dati in tqdm(d_train):
        
        o1.zero_grad()
        
        ged_t = []
        ged_r = []
        y_r = []
        y_p = []

        
        for pivot in dati:
            
            g1, g2, mask = pivot
            
            ged_p, _ = model(g1.to(device), g2.to(device), mask.to(device), True, smooth=False)
            
            ged_t.append(ged_p.unsqueeze(-1))
            ged_r.append(abs(g1.y-g2.y))
            y_r.append(g1.y)
            y_p.append(g2.y)
        
        ged_t = torch.cat(ged_t, -1)
        ged_r = torch.cat(ged_r, -1)
        ged_rp = ged_r.view(ged_t.size(0), 2 , -1)
        
        ged_rp = ged_rp.mean(-1)
        
        pos_indices = torch.argmin(ged_rp, -1).unsqueeze(1)

        log_probs = F.log_softmax(-ged_t / 2, dim=1)
        kl_loss = -log_probs.gather(1, pos_indices).squeeze(1)
        kl_loss = kl_loss.mean()


        y_r = torch.cat(y_r, -1).mean(-1)
        
        ged_tp = ged_t.view(ged_t.size(0), 2 , -1)
        ged_tp = ged_tp.mean(-1)
                
        y_pred = F.softmax(-ged_tp)
        y_r = torch.nn.functional.one_hot(y_r.long(), num_classes=2)
        
        #y_pred = classificator(ged_t).view(-1)
        loss_r = bceLoss(y_pred.float(), y_r.float())
        
        real_p.append(y_r)
        pred_p.append(y_pred)
                
        loss = kl_loss + loss_r 

        loss_ged_t += kl_loss
        loss_y_t += loss_r
        
        loss.backward()
        o1.step()

    loss_ged_t = loss_ged_t / len(d_train)
    loss_y_t = loss_y_t / len(d_train)

    real_p = torch.cat(real_p)
    pred_p = torch.cat(pred_p)

    return loss_ged_t, loss_y_t, real_p, pred_p


def evalBinary(model, classificator, d_train, temperatura=1):
    loss_ged_t = 0
    loss_y_t = 0

    real_p = []
    pred_p = []

    with torch.no_grad():
        for dati in tqdm(d_train):
            
            ged_t = []
            ged_r = []
            y_r = []
            y_p = []

            for pivot in dati:
                g1, g2, mask = pivot

                ged_p, _ = model(g1.to(device), g2.to(device), mask.to(device), True, smooth=False)

                ged_t.append(ged_p.unsqueeze(-1))
                ged_r.append(abs(g1.y - g2.y))
                y_r.append(g1.y)
                y_p.append(g2.y)

            ged_t = torch.cat(ged_t, -1)
            ged_r = torch.cat(ged_r, -1)
            ged_rp = ged_r.view(ged_t.size(0), 2, -1)

            ged_rp = ged_rp.mean(-1)
            pos_indices = torch.argmin(ged_rp, -1).unsqueeze(1)

            log_probs = F.log_softmax(-ged_t / temperatura, dim=1)
            kl_loss = -log_probs.gather(1, pos_indices).squeeze(1)
            kl_loss = kl_loss.mean()

            y_r = torch.cat(y_r, -1).mean(-1)

            y_pred = classificator(ged_t).view(-1)

            loss_r = bceLoss(y_pred, y_r)

            real_p.append(y_r)
            pred_p.append(y_pred)

            loss_ged_t += kl_loss
            loss_y_t += loss_r

    loss_ged_t = loss_ged_t / len(d_train)
    loss_y_t = loss_y_t / len(d_train)

    real_p = torch.cat(real_p)
    pred_p = torch.cat(pred_p)

    return loss_ged_t, loss_y_t, real_p, pred_p


def getPerformanceG(epoch, loss_ged_t, loss_ged_v,
                    real_y, pred_y, real_y_v, pred_y_v,
                    loss_y_t = None, loss_y_v = None, regression = True
                    ):

    real_y = real_y.detach().cpu().numpy()
    pred_y = pred_y.detach().cpu().numpy()

    real_y_v = real_y_v.detach().cpu().numpy()
    pred_y_v = pred_y_v.detach().cpu().numpy()

    print()
    print(f"Epoch {epoch}")
    print(f"Train Contr: {loss_ged_t:.3f}\tTest Contr: {loss_ged_v:.3f}")
    print(f"Train Y:     {loss_y_t:.3f}\tTest Y:     {loss_y_v:.3f}")
    if regression:
        print(f"Train R2:    {r2_score(real_y, pred_y):.3f}\tTest R2:    {r2_score(real_y_v, pred_y_v):.3f}")
    else:
        print(f"Train ROC:   {roc_auc_score(real_y, pred_y):.3f}\tTest ROC:   {roc_auc_score(real_y_v, pred_y_v):.3f}")
