import torch
from torch import nn

class Similarity(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.temp = temp
        self.cos = nn.CosineSimilarity(dim=-1)

    def forward(self, x, y):
        return self.cos(x, y) / self.temp
    
class Simcse(nn.Module):
    def __init__(self, temp):
        super().__init__()
        self.distance_metric = Similarity(temp)

    def forward(self, embs, labels):
        indexes = torch.randperm(embs.shape[0]).to('cuda')
        distance_matrix = self.distance_metric(embs, embs[indexes,:]).unsqueeze(1)
        condition = torch.where(labels==labels[indexes], 1 ,0)
        distance_matrix = torch.cat([distance_matrix, 1-distance_matrix], dim=1)
        loss = nn.functional.cross_entropy(distance_matrix, condition)
        return loss
        