import torch
import torch.nn as nn
from torch.nn import functional as F

def stable_normalization(P, num_iters=20, reg=0.1, epsilon=1e-8, convergence_tol=1e-6):
    K = torch.exp(-P / reg)

    for i in range(num_iters):
        prev_P = K.clone()
        row_sums = P.sum(dim=-1, keepdim=True)
        P = P / torch.where(row_sums > epsilon, row_sums, torch.ones_like(row_sums))

        col_sums = P.sum(dim=-2, keepdim=True)
        P = P / torch.where(col_sums > epsilon, col_sums, torch.ones_like(col_sums))
        P = torch.clamp(P, min=epsilon, max=1.0)

        diff = torch.norm(P - prev_P)

        if diff < convergence_tol:
            break

    return P

class GumbelSinkhornModel(nn.Module):

    def __init__(self, num_iters=20, init_temperature=1.0, final_temperature=0.1, noise_factor=0.015, num_epochs=100):

        super(GumbelSinkhornModel, self).__init__()
        self.num_iters = num_iters
        self.init_temperature = init_temperature
        self.final_temperature = final_temperature

        self.step_ = abs(init_temperature - final_temperature) / num_epochs

        self.noise_factor = noise_factor

    def concrete_sample(self, logits, tau):

        gumbel_noise = self._sample_gumbel(logits.size()).to(logits.device)

        y = (logits + gumbel_noise) / tau
        return F.softmax(y, dim=-1)

    def _sample_gumbel(self, shape, eps=1e-20):

        U = torch.rand(shape)
        return -torch.log(-torch.log(U + eps) + eps)

    def gumbel_sinkhorn(self, logits, steps=0):

        batch_size, n, _ = logits.shape

        temperature = max(self.init_temperature * (0.9999 ** steps), self.final_temperature)

        gumbel_noise = self.concrete_sample(logits, temperature).to(logits.device)
        gumbel_noise_back = self.concrete_sample(logits, 0.99).to(logits.device)

        noisy_logits = logits + self.noise_factor * gumbel_noise + ((self.noise_factor * gumbel_noise_back) -
                                                                    (self.noise_factor * gumbel_noise_back).detach())

        P = F.softmax(noisy_logits, dim=-1)

        P = stable_normalization(P, num_iters=150, epsilon=1e-20, convergence_tol=1e-3)

        return P

    def forward(self, logits, steps=0):
        perm_matrix = self.gumbel_sinkhorn(logits, steps=steps)

        return perm_matrix

class GumbelSinkhornTrainableModel(nn.Module):
    def __init__(self, num_iters=20, init_temperature=1.0, final_temperature=0.1, noise_factor=0.05, input_dim=5,
                 num_epochs=100):
        super(GumbelSinkhornTrainableModel, self).__init__()
        self.gumbel_sinkhorn = GumbelSinkhornModel(num_iters, init_temperature, final_temperature, noise_factor,
                                                   num_epochs)

        self.learning = nn.Parameter(torch.eye(input_dim, input_dim) * 0.99)

        self.norm = nn.LayerNorm(input_dim * input_dim)

    def forward(self, cost_matrix, steps):

        batch_size, n, _ = cost_matrix.shape

        logits = self.learning * cost_matrix

        logits = self.norm(logits.view(batch_size, -1)).view(batch_size, n, n)

        perm_matrix = self.gumbel_sinkhorn(logits, steps)

        total_cost = torch.sum(perm_matrix * cost_matrix, dim=[1, 2])

        return total_cost, perm_matrix
