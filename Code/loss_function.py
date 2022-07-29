import torch
import torch.nn as nn


class MyTriplet_loss(nn.Module):
    def __init__(self, margin=1.0, loss_weight=1.0):
        super(MyTriplet_loss, self).__init__()
        self.margin = margin
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.loss_weight = loss_weight

    def forward(self, inputs, targets):
        # distances
        n = inputs.size(0)
        dist = torch.pow(inputs, 2).sum(dim=1, keepdim=True).expand(n, n)
        dist = dist + dist.t()
        dist.addmm_(inputs, inputs.t(), beta=1, alpha=-2)
        dist = dist.clamp(min=1e-16).sqrt()
        ap_dist = dist.unsqueeze(2)
        an_dist = dist.unsqueeze(1)
        triplet_loss = ap_dist - an_dist + self.margin
        # triplets mask
        mask = get_mask(targets)
        # loss
        triplet_loss = torch.multiply(mask, triplet_loss)
        triplet_loss = torch.maximum(triplet_loss, torch.tensor(0.0))
        num_triplets = torch.sum(torch.tensor(torch.greater(triplet_loss, 1e-16), dtype=torch.float32))
        triplet_loss = torch.sum(triplet_loss) / (num_triplets + 1e-16)

        return triplet_loss


def get_mask(targets):
    indices = torch.logical_not(torch.tensor(torch.eye(targets.shape[0]), dtype=torch.bool).cuda())

    i_j = indices.unsqueeze(2)
    i_k = indices.unsqueeze(1)
    j_k = indices.unsqueeze(0)

    dist_indices = torch.logical_and(torch.logical_and(i_j, i_k), j_k)

    targets_equal = targets.unsqueeze(0).eq(targets.unsqueeze(1))
    i_equal_j = targets_equal.unsqueeze(2)
    i_equal_k = targets_equal.unsqueeze(1)
    valid_labels = torch.logical_and(i_equal_j, torch.logical_not(i_equal_k))

    mask = torch.logical_and(valid_labels, dist_indices)

    return torch.tensor(mask, dtype=torch.float32)