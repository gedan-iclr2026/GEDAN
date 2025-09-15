import torch
import numpy as np
import torch.nn as nn

from torch.functional import F
from torch_geometric.nn import SimpleConv, GINConv

device = "cuda"
torch.manual_seed(42)

if torch.cuda.is_available():
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)

np.random.seed(42)

# Cosine Similarity
cos_d = nn.CosineSimilarity()

# F cost function
class F_cost(torch.nn.Module):
    def __init__(self, feat):
        super().__init__()
        self.feat = feat
        self.fc1 = nn.Linear(feat*2, feat)
        self.fc2 = nn.Linear(feat, 1)
        self.normc = nn.BatchNorm1d(feat*2)

    def forward(self, a, b):

        c = torch.cat((a, b), -1)
        c = self.normc(c)

        x = self.fc1(c)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.softplus(x, beta=5)

        return x

# Main Model
class GEDAN(torch.nn.Module):

    def __init__(self, input_size, latent_space,  n_layer, max_node, sinkhorn, train_weight=True, diz_w=None, scale_p=1, lc = False):
        super().__init__()

        if diz_w is None:
            diz_w = [0.99, 0.99, 0.99, 0.99, 0.99]

        para_w = (diz_w[1]+diz_w[2])/2

        l_space = latent_space
        self.feat = input_size
        self.n_layer = n_layer
        self.max_node = max_node
        self.sinkhorn = sinkhorn
        self.train_weight = train_weight

        self.lambda_reg = 0.0001
        self.learned_cost = lc

        for param in self.sinkhorn.parameters():
            param.requires_grad = False

        if train_weight:
            self.weight_add = nn.Parameter(torch.tensor(diz_w[2]), requires_grad=False)
            self.weight_rem = nn.Parameter(torch.tensor(diz_w[1]), requires_grad=False)

            self.weight_sub1 = nn.Parameter(torch.tensor(diz_w[0]), requires_grad=False)
            self.weight_sub2 = nn.Parameter(torch.tensor(diz_w[0]), requires_grad=False)
            self.weight_sub3 = nn.Parameter(torch.tensor(diz_w[0]), requires_grad=False)

            self.weight_subA1 = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.weight_subA2 = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.weight_subA3 = nn.Parameter(torch.zeros(1), requires_grad=False)

            self.weight_edge1 = nn.Parameter(torch.tensor(diz_w[4]), requires_grad=False)
            self.weight_edge2 = nn.Parameter(torch.tensor(diz_w[3]), requires_grad=False)

            self.weight_edge2A = nn.Parameter(torch.tensor(diz_w[4]), requires_grad=False)
            self.weight_edge2B = nn.Parameter(torch.tensor(diz_w[3]), requires_grad=False)

            self.w_max = nn.Parameter(torch.tensor(200.0), requires_grad=False)

        else:
            self.weight_add = nn.Parameter(torch.tensor(diz_w[2]), requires_grad=False)
            self.weight_rem = nn.Parameter(torch.tensor(diz_w[1]), requires_grad=False)

            self.weight_sub1 = nn.Parameter(torch.tensor(diz_w[0]), requires_grad=False)
            self.weight_sub2 = nn.Parameter(torch.tensor(diz_w[0]), requires_grad=False)
            self.weight_sub3 = nn.Parameter(torch.tensor(diz_w[0]), requires_grad=False)

            self.weight_edge1 = nn.Parameter(torch.tensor(diz_w[4]), requires_grad=False)
            self.weight_edge2 = nn.Parameter(torch.tensor(diz_w[3]), requires_grad=False)

            self.weight_subA1 = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.weight_subA2 = nn.Parameter(torch.zeros(1), requires_grad=False)
            self.weight_subA3 = nn.Parameter(torch.zeros(1), requires_grad=False)

            self.weight_edge1 = nn.Parameter(torch.tensor(diz_w[4]), requires_grad=False)
            self.weight_edge2 = nn.Parameter(torch.tensor(diz_w[3]), requires_grad=False)

            self.weight_edge2A = nn.Parameter(torch.tensor(diz_w[4]), requires_grad=False)
            self.weight_edge2B = nn.Parameter(torch.tensor(diz_w[3]), requires_grad=False)

            self.w_max = nn.Parameter(torch.tensor(200.0), requires_grad=False)

        def custom_weight_init(m):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                try:
                    nn.init.normal_(m.bias)
                except:
                    None

        self.enc = nn.Sequential(nn.Linear(input_size, 16), nn.LeakyReLU(), nn.LayerNorm(16))

        self.enc.apply(custom_weight_init)

        self.param_cost_a = nn.Parameter(torch.zeros(1), requires_grad=True)

        self.gin_0 = GINConv(
            nn.Sequential(nn.Linear(input_size, l_space // 2),
                          nn.ReLU(),
                          nn.Linear(l_space // 2, l_space),
                          nn.LayerNorm(l_space)
                          ).apply(custom_weight_init)
        )

        self.gin_1 = GINConv(
            nn.Sequential(nn.Linear(l_space, l_space // 2),
                          nn.ReLU(),
                          nn.Linear(l_space // 2, l_space),
                          nn.LayerNorm(l_space)
                          ).apply(custom_weight_init)
        )

        self.message = SimpleConv(aggr="mean")

        self.gin_e = nn.ModuleList([GINConv(nn.Sequential(nn.Linear(l_space, l_space // 2),
                                            nn.ReLU(),
                                            nn.Linear(l_space // 2, l_space),
                                            nn.LayerNorm(l_space)
                                            )).to(device) for _ in range(n_layer)])

        self.norm = nn.SyncBatchNorm(max_node)
        self.normC = nn.SyncBatchNorm(max_node)
        self.norm1 = nn.BatchNorm1d(1)

        self.delta1 = nn.Parameter(torch.ones(1), requires_grad=True)
        self.delta2 = nn.Parameter(torch.ones(1), requires_grad=True)

        self.den = nn.Parameter(torch.ones(1)*(n_layer+2)*para_w*scale_p, requires_grad=True)

        if train_weight:
            self.costN0 = F_cost(input_size)
            self.costN1 = F_cost(l_space)
            self.costN2 = F_cost(l_space)

            self.cost_E = nn.ModuleList([F_cost(l_space).apply(custom_weight_init).to(device) for _ in range(n_layer+2)])
            self.weight_E = [nn.Parameter(torch.rand(1), requires_grad=True).to(device) for _ in range(n_layer+2)]

            self.costN0.apply(custom_weight_init)
            self.costN1.apply(custom_weight_init)
            self.costN2.apply(custom_weight_init)


    def getDiffNode(self, x, model_n = None):
        g1 = x[:, :, :, 0, :]
        g2 = x[:, :, :, 1, :]

        g1_ = g1.view(-1, g1.size(-1))
        g2_ = g2.view(-1, g2.size(-1))

        g1_ = F.normalize(g1_, p=2)
        g2_ = F.normalize(g2_, p=2)

        dist = abs(1-((cos_d(g1_, g2_)+1)/2)).view(x.size(0), x.size(1), x.size(2))
        if model_n is None:
            return dist
        else:
            cost = model_n.forward(g1_, g2_).view(g1.size(0), g1.size(1), g1.size(2))
            return dist, cost


    def getDiffNode1(self, x, e, model_n=None):

        g1 = x[:, :, :, 0, :]
        g2 = x[:, :, :, 1, :]

        e1 = e[:, :, :, 0, :]
        e2 = e[:, :, :, 1, :]

        g1_ = g1.view(-1, g1.size(-1))
        g2_ = g2.view(-1, g2.size(-1))

        e1_ = e1.view(-1, e1.size(-1))
        e2_ = e2.view(-1, e2.size(-1))

        g1_ = F.normalize(g1_, p=2)
        g2_ = F.normalize(g2_, p=2)

        dist = abs(1 - ((cos_d(g1_, g2_) + 1) / 2)).view(x.size(0), x.size(1), x.size(2))

        dist_e1 = F.relu(e1_-e2_).sum(-1).view(x.size(0), x.size(1), x.size(2))
        dist_e2 = F.relu(e2_-e1_).sum(-1).view(x.size(0), x.size(1), x.size(2))

        dist_e1 = F.normalize(dist_e1, p=2)*self.weight_edge2
        dist_e2 = F.normalize(dist_e2, p=2)*self.weight_edge1

        if model_n is not None:
            cost = model_n.forward(g1_, g2_).view(g1.size(0), g1.size(1), g1.size(2))
            return dist, dist_e1, dist_e2, cost
        else:
            return dist, dist_e1, dist_e2

    def getDiffDegree(self, x):
        g1 = x[:, :, :, 0, :]
        g2 = x[:, :, :, 1, :]

        return F.relu(g1 - g2).sum(-1), F.relu(g2 - g1).sum(-1)

    def getPair(self, g1_tmp, g2_tmp):
        g1_tmp_expanded = g1_tmp[:, None, :].expand(-1, g2_tmp.size(0), -1)
        g2_tmp_expanded = g2_tmp[None, :, :].expand(g1_tmp.size(0), -1, -1)

        matrix_concat = torch.stack([g1_tmp_expanded, g2_tmp_expanded], dim=2)

        matrix_concat = matrix_concat.unsqueeze(0)

        return matrix_concat

    def getPairBatch(self, g1x, g2x, batch):
        u_batch = torch.unique_consecutive(batch)

        graph = []

        for u in u_batch:
            mask = batch == u

            g1_tmp = g1x[mask]
            g2_tmp = g2x[mask]

            matrix_concat = self.getPair(g1_tmp, g2_tmp)
            graph.append(matrix_concat)

        graph = torch.cat(graph)

        return graph

    def freezeWeightGeneral(self, freeze):
        self.weight_add.requires_grad_(freeze)
        self.weight_rem.requires_grad_(freeze)

        self.weight_sub1.requires_grad_(freeze)
        self.weight_sub2.requires_grad_(freeze)
        self.weight_sub3.requires_grad_(freeze)

        self.weight_subA1.requires_grad_(freeze)
        self.weight_subA2.requires_grad_(freeze)
        self.weight_subA3.requires_grad_(freeze)

        self.weight_edge1.requires_grad_(freeze)
        self.weight_edge2.requires_grad_(freeze)

        self.weight_edge2A.requires_grad_(freeze)
        self.weight_edge2B.requires_grad_(freeze)
        self.w_max.requires_grad_(freeze)

    def freezeWeight(self, freeze):

        if not freeze:

            self.weight_add.requires_grad_(False)
            self.weight_rem.requires_grad_(False)

            self.weight_sub2.requires_grad_(False)
            self.weight_sub1.requires_grad_(False)

            self.weight_edge1.requires_grad_(False)
            self.weight_edge2.requires_grad_(False)

            self.norm.requires_grad_(False)
            self.den.requires_grad_(True)

            self.enc.requires_grad_(False)
            self.gin_0.requires_grad_(True)
            [gg.requires_grad_(True) for gg in self.gin_e]

        else:

            self.weight_add.requires_grad_(False)
            self.weight_rem.requires_grad_(False)

            self.weight_sub2.requires_grad_(False)
            self.weight_sub1.requires_grad_(False)

            self.weight_edge1.requires_grad_(False)
            self.weight_edge2.requires_grad_(False)

            self.norm.requires_grad_(False)

            self.freezeWeightGeneral(True)

            self.den.requires_grad_(True)
            self.gin_0.requires_grad_(True)
            [gg.requires_grad_(True) for gg in self.gin_e]

            self.enc.requires_grad_(False)
            self.costN0.requires_grad_(True)
            self.costN1.requires_grad_(True)
            [ce.requires_grad_(True) for ce in self.cost_E]

    def forward(self, g1, g2, mask, supervisioned, smooth=False):

        if supervisioned:
            self.freezeWeight(supervisioned)
            return self.supervisionedGED(g1, g2, mask)
        else:
            self.freezeWeight(supervisioned)
            return self.unsupervisionedGED(g1, g2, mask)

    def geMatrixCost(self, diff_node_, edge1, edge2, mask):

        mask_sub = mask[:, 0]
        #mask_sub = mask_sub.swapaxes(1, 2) # Fix
        mask_add = mask[:, 1]
        mask_rem = mask[:, 2]
        mask_pad = mask[:, 3]

        if self.learned_cost:
            edge = (edge1 * F.softplus(self.weight_edge1, beta=5) +
                    edge2 * F.softplus(self.weight_edge2, beta=5)).swapaxes(1, 2)

            matrix_sub = ((edge + (diff_node_ * F.softplus(self.weight_sub1, beta=5)))
                          * F.softplus(self.weight_sub2, beta=5)) * mask_sub.swapaxes(1, 2)

            matrix_add = (edge + F.softplus(self.weight_add, beta=5)) * mask_add
            matrix_rmv = (edge + F.softplus(self.weight_rem, beta=5)) * mask_rem
        else:

            edge = (edge1 *self.weight_edge1 +
                    edge2 *self.weight_edge2).swapaxes(1, 2)

            matrix_sub = ((edge + (diff_node_ * self.weight_sub1))
                          * self.weight_sub2) * mask_sub.swapaxes(1, 2)

            matrix_add = (edge + self.weight_add) * mask_add
            matrix_rmv = (edge + self.weight_rem) * mask_rem


        mask_pad_ = mask_pad + mask_add + mask_sub + mask_rem
        mask_pad_ = ~mask_pad_

        matrix_sub = matrix_sub.swapaxes(1, 2)
        all_matrix = (matrix_add + matrix_rmv + matrix_sub)
        mask_pad_ = ((all_matrix.max() + F.relu(self.w_max)) * mask_pad_)
        all_matrix = all_matrix + mask_pad_

        
        return all_matrix

    def unsupervisionedGED(self, g1, g2, mask):

            graph_x = self.getPairBatch(g1.x, g2.x, g1.batch)
            graph_d = self.getPairBatch(g1.dg, g2.dg, g1.batch)

            diff_node = self.getDiffNode(graph_x).transpose(1, 2)
            diff_node[diff_node > 0] = 1

            x1 = self.gin_0(g1.x, g1.edge_index)
            x2 = self.gin_0(g2.x, g2.edge_index)

            edge_1 = self.message(g1.x, g1.edge_index)
            edge_2 = self.message(g2.x, g2.edge_index)

            edge1, edge2 = self.getDiffDegree(graph_d)

            graph_x1 = self.getPairBatch(x1, x2, g1.batch)
            graph_e1 = self.getPairBatch(edge_1, edge_2, g1.batch)

            diff_node_1, e1, e2 = self.getDiffNode1(graph_x1, graph_e1)
            diff_node_1 = diff_node_1.transpose(1, 2)
            diff_node_1[diff_node_1 > 0] = diff_node_1[diff_node_1 > 0] + ((e1 + e2))[diff_node_1 > 0]

            diff_node_ = diff_node + diff_node_1

            for ij in range(self.n_layer):
                x1 = self.gin_e[ij](x1, g1.edge_index)+x1
                x2 = self.gin_e[ij](x2, g2.edge_index)+x2

                edge_1 = self.message(edge_1, g1.edge_index)
                edge_2 = self.message(edge_2, g2.edge_index)

                graph_e2 = self.getPairBatch(edge_1, edge_2, g1.batch)

                graph_x2 = self.getPairBatch(x1, x2, g1.batch)

                diff_node_2, e1, e2 = self.getDiffNode1(graph_x2, graph_e2)
                diff_node_2 = diff_node_2.transpose(1, 2)

                diff_node_2[diff_node_2 > 0] = diff_node_2[diff_node_2 > 0] + ((e1+e2))[diff_node_2 > 0]

                diff_node_ = diff_node_ + diff_node_2

            all_matrix = self.geMatrixCost(diff_node_/self.den,
                                           edge1,
                                           edge2,
                                           mask)

            all_n = self.norm(all_matrix)
            ged0, perm_matrix = self.sinkhorn(all_n, steps=1)

            matrice = perm_matrix * all_matrix
            ged0 = torch.sum(matrice, dim=[1, 2]) + torch.log(self.den)

            return ged0, ged0

    def supervisionedGED(self, g1, g2, mask):

            graph_x = self.getPairBatch(g1.x, g2.x, g1.batch)
            graph_d = self.getPairBatch(g1.dg, g2.dg, g1.batch)

            diff_node, cost_0 = self.getDiffNode(graph_x, self.costN0)
            diff_node = diff_node.transpose(1, 2)
            diff_node[diff_node > 0] = 1

            diff_cost = (diff_node * cost_0.transpose(1, 2) * self.weight_E[0])

            x1 = self.gin_0(g1.x, g1.edge_index)
            x2 = self.gin_0(g2.x, g2.edge_index)

            edge_1 = self.message(g1.x, g1.edge_index)
            edge_2 = self.message(g2.x, g2.edge_index)

            edge1, edge2 = self.getDiffDegree(graph_d)

            graph_x1 = self.getPairBatch(x1, x2, g1.batch)
            graph_e1 = self.getPairBatch(edge_1, edge_2, g1.batch)

            diff_node_1, e1, e2, cost_1 = self.getDiffNode1(graph_x1, graph_e1, self.costN1)
            diff_node_1 = diff_node_1.transpose(1, 2)

            diff_cost_1 = (diff_node_1 * cost_1.transpose(1, 2) * self.weight_E[1])

            diff_node_ = diff_node + diff_node_1
            diff_cost_ = diff_cost + diff_cost_1

            for ij in range(self.n_layer):

                x1 = self.gin_e[ij](x1, g1.edge_index)+x1
                x2 = self.gin_e[ij](x2, g2.edge_index)+x2

                edge_1 = self.message(edge_1, g1.edge_index)
                edge_2 = self.message(edge_2, g2.edge_index)

                graph_e2 = self.getPairBatch(edge_1, edge_2, g1.batch)

                graph_x2 = self.getPairBatch(x1, x2, g1.batch)

                diff_node_2, e1, e2, cost_2 = self.getDiffNode1(graph_x2, graph_e2, self.cost_E[ij])
                diff_node_2 = diff_node_2.transpose(1, 2)

                diff_cost_2 = (diff_node_2 * cost_2.transpose(1, 2) * self.weight_E[ij+2])

                diff_node_ = diff_node_ + diff_node_2
                diff_cost_ = diff_cost + diff_cost_2


            pesi = torch.sum(torch.cat(self.weight_E))

            all_matrix = self.geMatrixCost(diff_node_/self.den,
                                           edge1,
                                           edge2,
                                           mask)


            all_n = self.norm(all_matrix)

            ged0, perm_matrix = self.sinkhorn(all_n, steps=1)
            matrice = perm_matrix * (F.sigmoid(self.delta1)*all_matrix+
                                     F.sigmoid(self.delta2)*(diff_cost_/(pesi)))

            ged0 = torch.sum(matrice, dim=[1, 2]) + torch.log(self.den) * self.lambda_reg

            return ged0, ged0


    def getMatchAll(self, g1, g2, mask):

            graph_x = self.getPairBatch(g1.x, g2.x, g1.batch)
            graph_d = self.getPairBatch(g1.dg, g2.dg, g1.batch)

            diff_node, cost_0 = self.getDiffNode(graph_x, self.costN0)
            diff_node = diff_node.transpose(1, 2)
            diff_node[diff_node > 0] = 1

            diff_cost = (diff_node * cost_0.transpose(1, 2) * F.sigmoid(self.weight_E[0]))

            x1 = self.gin_0(g1.x, g1.edge_index)
            x2 = self.gin_0(g2.x, g2.edge_index)

            edge_1 = self.message(g1.x, g1.edge_index)
            edge_2 = self.message(g2.x, g2.edge_index)

            edge1, edge2 = self.getDiffDegree(graph_d)

            x_level = x1.unsqueeze(0)


            cost_level = self.geMatrixCost(diff_node,
                                           edge1,
                                           edge2,
                                           mask)

            cost_level = (F.sigmoid(self.delta1)*cost_level+ F.sigmoid(self.delta2)*diff_cost)


            graph_x1 = self.getPairBatch(x1, x2, g1.batch)
            graph_e1 = self.getPairBatch(edge_1, edge_2, g1.batch)

            diff_node_1, e1, e2, cost_1 = self.getDiffNode1(graph_x1, graph_e1, self.costN1)
            diff_node_1 = diff_node_1.transpose(1, 2)

            diff_cost_1 = (diff_node_1 * cost_1.transpose(1, 2) * F.sigmoid(self.weight_E[1]))

            diff_node_ = diff_node + diff_node_1
            diff_cost_ = diff_cost + diff_cost_1

            cost_level_1 = self.geMatrixCost(diff_node_1,
                                           edge1,
                                           edge2,
                                           mask)

            cost_level_1 = (F.sigmoid(self.delta1) * cost_level_1 + F.sigmoid(self.delta2) * diff_cost_1)

            cost_level_all = torch.cat((cost_level, cost_level_1), 0)

            for ij in range(self.n_layer):
                x1 = self.gin_e[ij](x1, g1.edge_index)+x1
                x2 = self.gin_e[ij](x2, g2.edge_index)+x2

                x_level = torch.cat((x_level, x1.unsqueeze(0)), 0)

                edge_1 = self.message(edge_1, g1.edge_index)
                edge_2 = self.message(edge_2, g2.edge_index)

                graph_e2 = self.getPairBatch(edge_1, edge_2, g1.batch)

                graph_x2 = self.getPairBatch(x1, x2, g1.batch)

                diff_node_2, e1, e2, cost_2 = self.getDiffNode1(graph_x2, graph_e2, self.cost_E[ij])
                diff_node_2 = diff_node_2.transpose(1, 2)

                diff_cost_2 = (diff_node_2 * cost_2.transpose(1, 2) * F.sigmoid(self.weight_E[ij+2]))

                diff_node_ = diff_node_ + diff_node_2
                diff_cost_ = diff_cost + diff_cost_2

                cost_level_2 = self.geMatrixCost(diff_node_2,
                                                 edge1,
                                                 edge2,
                                                 mask)

                cost_level_2 = (F.sigmoid(self.delta1) * cost_level_2 + F.sigmoid(self.delta2) * diff_cost_2)

                cost_level_all = torch.cat((cost_level_all, cost_level_2), 0)


            pesi = torch.sum(F.sigmoid(torch.cat(self.weight_E)))

            all_matrix = self.geMatrixCost(diff_node_/self.den,
                                           edge1,
                                           edge2,
                                           mask)

            all_n = self.norm(all_matrix)
            ged0, perm_matrix = self.sinkhorn(all_n, steps=1)
            matrice = perm_matrix * (F.sigmoid(self.delta1)*all_matrix+
                                     F.sigmoid(self.delta2)*(diff_cost_/(pesi)))


            ged0 = torch.sum(matrice, dim=[1, 2]) + torch.log(self.den) * self.lambda_reg

            return ged0, ged0, perm_matrix, cost_level_all, x_level, self.weight_E


# Second Loss Network
class ClassModelRegression(torch.nn.Module):

    def __init__(self, init_val, output_y, classification = False):
        super().__init__()

        self.classification = classification

        if classification:
            self.lin1 = nn.Linear(2, output_y)
        else:
            self.lin1 = nn.Linear(init_val, output_y)

        self.norm = nn.SyncBatchNorm(init_val)
        self.nn0 = nn.LayerNorm(init_val)


    def forward(self, ged_t):
        ged_t = self.nn0(ged_t)

        if self.classification:
            ged_t = ged_t.view(len(ged_t), 2, -1)
            ged_t = ged_t.mean(2)

        x = self.lin1(ged_t)

        return x
