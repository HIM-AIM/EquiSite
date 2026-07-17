import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn


def focal_loss(labels, logits, alpha, gamma):
    """Compute the focal loss between `logits` and the ground truth `labels`.

    Focal loss = -alpha_t * (1-pt)^gamma * log(pt)
    where pt is the probability of being classified to the true class.
    pt = p (if true class), otherwise pt = 1 - p. p = sigmoid(logit).

    Args:
        labels: A float tensor of size [batch, num_classes].
        logits: A float tensor of size [batch, num_classes].
        alpha: A float tensor of size [batch_size] specifying per-example
            weight for balanced cross entropy.
        gamma: A float scalar modulating loss from hard and easy examples.

    Returns:
        focal_loss: A float32 scalar representing normalized total loss.
    """
    BCLoss = F.binary_cross_entropy_with_logits(
        input=logits, target=labels, reduction="none"
    )
    # if gamma == 0.0:
    #     modulator = 1.0
    # else:
    #     modulator = torch.exp(-gamma * labels * logits - gamma * torch.log(
    #         1 + torch.exp(-1.0 * logits)))
    pt = torch.exp(-BCLoss)
    # loss = modulator * BCLoss
    F_loss = (1 - pt) ** gamma * BCLoss
    weighted_loss = alpha * F_loss
    focal_loss = torch.sum(weighted_loss)
    focal_loss /= torch.sum(labels)
    return focal_loss


def CB_loss(
    labels,
    logits,
    samples_per_cls,
    no_of_classes,
    loss_type,
    beta,
    gamma,
):
    """Compute the Class Balanced Loss between logits and labels.

    Class Balanced Loss: ((1-beta)/(1-beta^n))*Loss(labels, logits), where
    Loss is one of the standard losses used for neural networks.
    """
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    labels_one_hot = F.one_hot(
        labels.squeeze().to(torch.int64), no_of_classes
    ).float()
    weights = torch.tensor(weights).float().to("cuda")
    weights = weights.unsqueeze(0)
    weights = weights.repeat(labels_one_hot.shape[0], 1) * labels_one_hot
    weights = weights.sum(1)
    weights = weights.unsqueeze(1)
    weights = weights.repeat(1, no_of_classes)

    if loss_type == "focal":
        if len(labels_one_hot) != len(logits):
            print("error")
        cb_loss = focal_loss(labels_one_hot, logits, weights, gamma)
    elif loss_type == "sigmoid":
        cb_loss = F.binary_cross_entropy_with_logits(
            input=logits, target=labels_one_hot, weight=weights
        )
    elif loss_type == "softmax":
        pred = logits.softmax(dim=1)
        cb_loss = F.binary_cross_entropy(
            input=pred, target=labels_one_hot, weight=weights
        )
    return cb_loss



class CenterLoss(nn.Module):
    """Center loss.

    Reference:
        Wen et al. A Discriminative Feature Learning Approach for Deep Face
        Recognition. ECCV 2016.

    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """

    def __init__(self, num_classes=2, feat_dim=2048, use_gpu=True):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.use_gpu = use_gpu

        if self.use_gpu:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim).cuda()
            )
        else:
            self.centers = nn.Parameter(
                torch.randn(self.num_classes, self.feat_dim)
            )

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (num_classes).
        """
        assert x.size(0) == labels.size(0), (
            "features.size(0) is not equal to labels.size(0)"
        )
        batch_size = x.size(0)
        distmat = (
            torch.pow(x, 2)
            .sum(dim=1, keepdim=True)
            .expand(batch_size, self.num_classes)
            + torch.pow(self.centers, 2)
            .sum(dim=1, keepdim=True)
            .expand(self.num_classes, batch_size)
            .t()
        )
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.use_gpu:
            classes = classes.cuda()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))
        print(mask)

        dist = []
        for i in range(batch_size):
            print(mask[i])
            value = distmat[i][mask[i]]
            value = value.clamp(
                min=1e-12, max=1e+12
            )  # for numerical stability
            dist.append(value)
        dist = torch.cat(dist)
        loss = dist.mean()
        return loss


class TripletCenterLoss(nn.Module):
    def __init__(self, margin=5, num_classes=2, center_embed=2):
        super(TripletCenterLoss, self).__init__()
        self.margin = margin
        self.num_classes = num_classes
        self.ranking_loss = nn.MarginRankingLoss(margin=margin)
        self.centers = nn.Parameter(torch.randn(num_classes, center_embed))

    def forward(self, inputs, targets):
        targets = targets.view(-1).to(torch.int64)

        # Distance from every embedding to every learnable class center.
        distances = torch.cdist(inputs, self.centers, p=2)
        dist_ap = distances.gather(1, targets.unsqueeze(1)).squeeze(1)

        positive_mask = F.one_hot(
            targets, num_classes=self.num_classes
        ).bool()
        dist_an = distances.masked_fill(
            positive_mask, float("inf")
        ).min(dim=1).values

        # dist_an should be larger than dist_ap by at least margin.
        return self.ranking_loss(
            dist_an, dist_ap, torch.ones_like(dist_an)
        )


