import matplotlib.pyplot as plt
import numpy as np
import torch
import random

from tqdm import tqdm
from sklearn.model_selection import GridSearchCV
from sklearn.decomposition import PCA, KernelPCA
from sklearn.manifold import TSNE

from sklearn.preprocessing import PowerTransformer, RobustScaler
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.svm import SVC, SVR
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neural_network import MLPClassifier, MLPRegressor

from sklearn.metrics import roc_auc_score, r2_score
from sklearn.metrics import mean_absolute_error

import warnings
warnings.filterwarnings("ignore")


def getPlot(dataset_n, ti, idx_,
            ugedan_x_train, ugedan_y_train,
            sgedan_x_train, sgedan_y_train,
            ugedan_x_test, ugedan_y_test,
            sgedan_x_test, sgedan_y_test,

            eg0_x_train, eg0_y_train,
            e0_x_train, e0_y_train,
            eg0_x_test, eg0_y_test,
            e0_x_test, e0_y_test,
            ):

    if ti == "KPCA":
        mds1 = KernelPCA(n_components=2, random_state=0)
        mds2 = KernelPCA(n_components=2, random_state=0)
    if ti == "PCA":
        mds1 = PCA(n_components=2, random_state=0)
        mds2 = PCA(n_components=2, random_state=0)
    if ti == "TNSE":
        perp = 30
        mds1 = TSNE(n_components=2, init="random", random_state=42, perplexity=perp)
        mds2 = TSNE(n_components=2, init="random", random_state=42, perplexity=perp)

    ssize = 5

    plt.figure(figsize=(14, 5))
    plt.subplot(2, 4, 1)
    plt.title("Train GEDAN - Grid Search")
    coordinates = mds1.fit_transform(ugedan_x_train)
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=ugedan_y_train, s = ssize, cmap="bwr")
    cbar = plt.colorbar(scatter)

    plt.subplot(2, 4, 2)
    plt.title("Train GMSE - e = 0")
    coordinates = mds1.fit_transform(e0_x_train)
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=e0_y_train, s = ssize, cmap="bwr")
    cbar = plt.colorbar(scatter)

    plt.subplot(2, 4, 3)
    plt.title("Train GMSE - e > 0")
    coordinates = mds1.fit_transform(eg0_x_train)
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=eg0_y_train, s = ssize, cmap="bwr")
    cbar = plt.colorbar(scatter)

    plt.subplot(2, 4, 4)
    plt.title("Train GEDAN - Learned")
    coordinates = mds2.fit_transform(sgedan_x_train)
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=sgedan_y_train, s = ssize, cmap="bwr")
    cbar = plt.colorbar(scatter)

    plt.subplot(2, 4, 5)
    plt.title("Test GEDAN - Grid Search")
    coordinates = mds1.transform(ugedan_x_test)
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=ugedan_y_test, s = ssize, cmap="bwr")
    cbar = plt.colorbar(scatter)

    plt.subplot(2, 4, 6)
    plt.title("Test GMSE - e = 0")
    coordinates = mds1.fit_transform(e0_x_test)
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=e0_y_test, s = ssize, cmap="bwr")
    cbar = plt.colorbar(scatter)

    plt.subplot(2, 4, 7)
    plt.title("Test GMSE - e > 0")
    coordinates = mds1.fit_transform(eg0_x_test)
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=eg0_y_test, s = ssize, cmap="bwr")
    cbar = plt.colorbar(scatter)

    plt.subplot(2, 4, 8)
    plt.title("Test GEDAN - Learned")
    coordinates = mds2.transform(sgedan_x_test)
    scatter = plt.scatter(coordinates[:, 0], coordinates[:, 1], c=sgedan_y_test, s = ssize, cmap="bwr")
    cbar = plt.colorbar(scatter)

    plt.tight_layout()
    plt.savefig(f"results/{dataset_n}/plot/{idx_}.svg")
    plt.close()


def getParam(model):

    if model == RandomForestRegressor:
        return {
            'n_estimators': [100],
            'max_depth': [10, 20],
        }

    if model == KNeighborsRegressor:
        return {
            'n_neighbors': [3, 5, 8, 16],
            'weights': ['uniform', 'distance'],
        }

    if model == MLPRegressor:
        return {
            'hidden_layer_sizes': [(128, 64), (64, 32)],
            'solver': ['adam'],
            'alpha': [0.001, 0.01],
            'max_iter': [200]
        }

    if model == SVR:
        return {
            'C': [1, 10, 100],
            'kernel': ["rbf"],
            'degree': [1, 2],
        }

    if model == RandomForestClassifier:
        return {
            'n_estimators': [100],
            'max_depth': [10, 20],
        }

    if model == KNeighborsClassifier:
        return {
            'n_neighbors': [3, 5, 8, 16],
            'weights': ['uniform', 'distance'],
        }

    if model == MLPClassifier:
        return {
            'hidden_layer_sizes': [(64, 32)],
            'solver': ['adam'],
            'alpha': [0.001, 0.01],
            'max_iter': [200]
        }

    if model == SVC:
        return {
            'C': [1, 10, 100],
            'kernel': ["rbf"],
            'degree': [1, 2],
        }


def testModel(data_train, label_train, data_test, label_test, model_c, params, metric_, trasf=None):

    stand = PowerTransformer()

    if trasf != None:
        data_train = stand.fit_transform(data_train)
        data_test = stand.transform(data_test)

    model_c = GridSearchCV(model_c, param_grid=params, scoring="neg_root_mean_squared_error", cv=5)

    model_c.fit(data_train, label_train)

    pred_test = model_c.predict(data_test)

    return metric_(label_test, pred_test)


def getLoad(type, dataset_name, pivot):
	
    if type == "UGEDAN":
        train_x = np.load(f"results/{dataset_name}/embedding/GEDAN/dati_UGEDAN_train_train_{pivot}.npy")
        train_y = np.load(f"results/{dataset_name}/embedding/GEDAN/label_UGEDAN_train_train_{pivot}.npy")

        test_x = np.load(f"results/{dataset_name}/embedding/GEDAN/dati_UGEDAN_test_train_{pivot}.npy")
        test_y = np.load(f"results/{dataset_name}/embedding/GEDAN/label_UGEDAN_test_train_{pivot}.npy")

    else:
        train_x = np.load(f"results/{dataset_name}/embedding/GEDAN/dati_SGEDAN_train_train_{pivot}.npy")
        train_y = np.load(f"results/{dataset_name}/embedding/GEDAN/label_SGEDAN_train_train_{pivot}.npy")

        test_x = np.load(f"results/{dataset_name}/embedding/GEDAN/dati_SGEDAN_test_train_{pivot}.npy")
        test_y = np.load(f"results/{dataset_name}/embedding/GEDAN/label_SGEDAN_test_train_{pivot}.npy")

    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    return train_x, train_y, test_x, test_y


def getLoadGMSM(type_, dataset_name, pivot):

    train_x = np.load(f"results/{dataset_name}/embedding/GMSM/{type_}_{dataset_name}_data_train_{pivot}.npy")
    train_y = np.load(f"results/{dataset_name}/embedding/GMSM/{type_}_{dataset_name}_label_train_{pivot}.npy")

    test_x = np.load(f"results/{dataset_name}/embedding/GMSM/{type_}_{dataset_name}_data_test_{pivot}.npy")
    test_y = np.load(f"results/{dataset_name}/embedding/GMSM/{type_}_{dataset_name}_label_test_{pivot}.npy")

    train_y = train_y.reshape(-1, 1)
    test_y = test_y.reshape(-1, 1)

    return train_x, train_y, test_x, test_y

def getPredictionEmbedding(dataset_name):

    seed = 42
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    classificazione = False

    if dataset_name == "FreeSolv":
        metrics = r2_score
        print("R2")

    if dataset_name == "BBBP":
        metrics = roc_auc_score
        print("ROC-AUC")
        classificazione = True


    u_m1, u_m2, u_m3, u_m4 = [], [], [], []
    s_m1, s_m2, s_m3, s_m4 = [], [], [], []
    eg_m1, eg_m2, eg_m3, eg_m4 = [], [], [], []
    e0_m1, e0_m2, e0_m3, e0_m4 = [], [], [], []

    for p in tqdm(range(3), desc="Pivots"):

        u_train_x, u_train_y, u_test_x, u_test_y = getLoad("UGEDAN", dataset_name, p)
        s_train_x, s_train_y, s_test_x, s_test_y = getLoad("SGEDAN", dataset_name, p)

        eg_train_x, eg_train_y, eg_test_x, eg_test_y = getLoadGMSM("EG0", dataset_name, p)
        e0_train_x, e0_train_y, e0_test_x, e0_test_y = getLoadGMSM("E0", dataset_name, p)

        stand = PowerTransformer()
        u_train_x = stand.fit_transform(u_train_x)
        u_test_x = stand.transform(u_test_x)

        stand = PowerTransformer()
        s_train_x = stand.fit_transform(s_train_x)
        s_test_x = stand.transform(s_test_x)

        stand = PowerTransformer()
        eg_train_x = stand.fit_transform(eg_train_x)
        eg_test_x = stand.transform(eg_test_x)

        stand = PowerTransformer()
        e0_train_x = stand.fit_transform(e0_train_x)
        e0_test_x = stand.transform(e0_test_x)


        getPlot(dataset_name, "KPCA", dataset_name+str(p),
                u_train_x, u_train_y,
                s_train_x, s_train_y,
                u_test_x, u_test_y,
                s_test_x, s_test_y,
                eg_train_x, eg_train_y,
                e0_train_x, e0_train_y,
                eg_test_x, eg_test_y,
                e0_test_x, e0_test_y
                )

        stand = RobustScaler()

        if classificazione:
            model_ = KNeighborsClassifier()
            param = getParam(KNeighborsClassifier)
        else:
            model_ = KNeighborsRegressor()
            param = getParam(KNeighborsRegressor)

        p1 = p2 = p3 = p4 = 0
        p1 = testModel(u_train_x, u_train_y, u_test_x, u_test_y, model_, param, metrics, trasf=stand)
        p2 = testModel(s_train_x, s_train_y, s_test_x, s_test_y, model_, param, metrics, trasf=stand)
        p3 = testModel(eg_train_x, eg_train_y, eg_test_x, eg_test_y, model_, param, metrics, trasf=stand)
        p4 = testModel(e0_train_x, e0_train_y, e0_test_x, e0_test_y, model_, param, metrics, trasf=stand)

        u_m1.append(p1)
        s_m1.append(p2)
        eg_m1.append(p3)
        e0_m1.append(p4)

        if classificazione:
            model_ = SVC()
            param = getParam(SVC)
        else:
            model_ = SVR()
            param = getParam(SVR)

        p1 = testModel(u_train_x, u_train_y, u_test_x, u_test_y, model_, param, metrics, trasf=stand)
        p2 = testModel(s_train_x, s_train_y, s_test_x, s_test_y, model_, param, metrics, trasf=stand)
        p3 = testModel(eg_train_x, eg_train_y, eg_test_x, eg_test_y, model_, param, metrics, trasf=stand)
        p4 = testModel(e0_train_x, e0_train_y, e0_test_x, e0_test_y, model_, param, metrics, trasf=stand)

        u_m2.append(p1)
        s_m2.append(p2)
        eg_m2.append(p3)
        e0_m2.append(p4)

        if classificazione:
            model_ = RandomForestClassifier(random_state=0)
            param = getParam(RandomForestClassifier)
        else:
            model_ = RandomForestRegressor(random_state=0)
            param = getParam(RandomForestRegressor)
        p1 = testModel(u_train_x, u_train_y, u_test_x, u_test_y, model_, param, metrics, trasf=stand)
        p2 = testModel(s_train_x, s_train_y, s_test_x, s_test_y, model_, param, metrics, trasf=stand)
        p3 = testModel(eg_train_x, eg_train_y, eg_test_x, eg_test_y, model_, param, metrics, trasf=stand)
        p4 = testModel(e0_train_x, e0_train_y, e0_test_x, e0_test_y, model_, param, metrics, trasf=stand)

        u_m3.append(p1)
        s_m3.append(p2)
        eg_m3.append(p3)
        e0_m3.append(p4)

        if classificazione:
            model_ = MLPClassifier(max_iter=1000, random_state=0)
            param = getParam(MLPClassifier)
        else:
            model_ = MLPRegressor(max_iter=1000, random_state=0)
            param = getParam(MLPRegressor)
        p1 = testModel(u_train_x, u_train_y, u_test_x, u_test_y, model_, param, metrics, trasf=stand)
        p2 = testModel(s_train_x, s_train_y, s_test_x, s_test_y, model_, param, metrics, trasf=stand)
        p3 = testModel(eg_train_x, eg_train_y, eg_test_x, eg_test_y, model_, param, metrics, trasf=stand)
        p4 = testModel(e0_train_x, e0_train_y, e0_test_x, e0_test_y, model_, param, metrics, trasf=stand)

        u_m4.append(p1)
        s_m4.append(p2)
        eg_m4.append(p3)
        e0_m4.append(p4)

        print(
            f"KNN\t{np.array(u_m1).mean():.3f}\t{np.array(eg_m1).mean():.3f}\t{np.array(e0_m1).mean():.3f}\t {np.array(s_m1).mean():.3f}")
        print(
            f"SVM\t{np.array(u_m2).mean():.3f}\t{np.array(eg_m2).mean():.3f}\t{np.array(e0_m2).mean():.3f}\t {np.array(s_m2).mean():.3f}")
        print(
            f"RF\t{np.array(u_m3).mean():.3f}\t{np.array(eg_m3).mean():.3f}\t{np.array(e0_m3).mean():.3f}\t {np.array(s_m3).mean():.3f}")
        print(
            f"MLP\t{np.array(u_m4).mean():.3f}\t{np.array(eg_m4).mean():.3f}\t{np.array(e0_m4).mean():.3f}\t {np.array(s_m4).mean():.3f}")
        print()

    print()

    print(f"\tGEDAN \tGMSM \tGMSM\t GEDAN")
    print(f"\tUnit \te>0 \te=0\tLearned")

    print(f"KNN\t{np.array(u_m1).mean():.3f}\t{np.array(eg_m1).mean():.3f}\t{np.array(e0_m1).mean():.3f}\t {np.array(s_m1).mean():.3f}")
    print(f"SVM\t{np.array(u_m2).mean():.3f}\t{np.array(eg_m2).mean():.3f}\t{np.array(e0_m2).mean():.3f}\t {np.array(s_m2).mean():.3f}")
    print(f"RF\t{np.array(u_m3).mean():.3f}\t{np.array(eg_m3).mean():.3f}\t{np.array(e0_m3).mean():.3f}\t {np.array(s_m3).mean():.3f}")
    print(f"MLP\t{np.array(u_m4).mean():.3f}\t{np.array(eg_m4).mean():.3f}\t{np.array(e0_m4).mean():.3f}\t {np.array(s_m4).mean():.3f}")


    u_metric = np.array(u_m1).mean()+np.array(u_m2).mean()+np.array(u_m3).mean()+np.array(u_m4).mean()
    s_metric = np.array(s_m1).mean()+np.array(s_m2).mean()+np.array(s_m3).mean()+np.array(s_m4).mean()
    eg_metric = np.array(eg_m1).mean()+np.array(eg_m2).mean()+np.array(eg_m3).mean()+np.array(eg_m4).mean()
    e0_metric = np.array(e0_m1).mean()+np.array(e0_m2).mean()+np.array(e0_m3).mean()+np.array(e0_m4).mean()

    u_metric = u_metric/4
    s_metric = s_metric/4
    eg_metric = eg_metric/4
    e0_metric = e0_metric/4
    print("----------------------------------------")
    print(f"Average\t{u_metric:.3f}\t{eg_metric:.3f}\t{e0_metric:.3f}\t {s_metric:.3f}")
