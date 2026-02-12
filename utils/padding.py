import torch
import torch.nn.functional as F

def sphere_padding(x, pad_list, l): #correct
    """
    :param x: [n, n_w]
    :param pad_list: [128, 64, 32]
    :param l: int
    :return:
    """
    j = l**2
    pad_idx = [k for k in range(l) for _ in range(2*(k+1) - 1)]
    w = int(x.shape[-1]/j)
    x = x.view(-1, j, w)
    xs = list(torch.split(x, 1, dim=1))
    for i, x in enumerate(xs):
        xs[i] = F.pad(x.squeeze(), (0, pad_list[pad_idx[i]]-w), 'constant', value=0)
    out_tensor = torch.cat(xs, dim=-1)
    return out_tensor
def fea1_sphere_padding(x, pad_list, l):
    """
    :param x: [n, n_w]
    :param pad_list: [128, 64, 32]
    :param l: int
    :return:
    """
    split_x = torch.split(x, 3, dim=-2)


    x = torch.cat(split_x, dim=-1)
    w = int(x.shape[-1])
    xs = list(torch.split(x, 1, dim=-2))
    for i, x in enumerate(xs):
        xs[i] = F.pad(x.squeeze(), (pad_list[i]*i, pad_list[i]*(i+1)-w), 'constant', value=0)
    out_tensor = torch.cat(xs, dim=-1)
    return out_tensor
