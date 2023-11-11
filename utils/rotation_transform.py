import torch

@torch.no_grad()
def rot_inner_all(x):
    num = x.shape[0]
    c, h, w = x.shape[1], x.shape[2], x.shape[3]

    R = x.repeat(4, 1, 1, 1)
    a = x.permute(0, 1, 3, 2)
    a = a.view(num, c, 2, h//2, w)

    a = a.permute(2, 0, 1, 3, 4)

    s1 = a[0]
    s2 = a[1]
    s1_1 = torch.rot90(s1, 2, (2, 3))
    s2_2 = torch.rot90(s2, 2, (2, 3))

    R[num:2 * num] = torch.cat((s1_1.unsqueeze(2), s2.unsqueeze(2)), dim=2).reshape(num, c, h, w).permute(0, 1, 3, 2)
    R[3 * num:] = torch.cat((s1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, c, h, w).permute(0, 1, 3, 2)
    R[2 * num:3 * num] = torch.cat((s1_1.unsqueeze(2), s2_2.unsqueeze(2)), dim=2).reshape(num, c, h, w).permute(0, 1, 3, 2)
    return R

@torch.no_grad()
def Rotation(x):
    # rotation augmentation in OCM
    X = rot_inner_all(x)
    return torch.cat((X, torch.rot90(X, 2, (2, 3)), torch.rot90(X, 1, (2, 3)), torch.rot90(X, 3, (2, 3))), dim=0)

@torch.no_grad()
def flip_inner(x, flip1, flip2):
    bsz, c, h, w = x.size()
    a = x
    a = a.view(bsz, c, 2, h//2, w)
    a = a.permute(2, 0, 1, 3, 4)
    s1 = a[0]
    s2 = a[1]
    if flip1:
        s1 = torch.flip(s1, (3,))
    if flip2:
        s2 = torch.flip(s2, (3,))
    s = torch.cat((s1.unsqueeze(2), s2.unsqueeze(2)), dim=2)
    S = s.reshape(bsz, c, h, w)
    return S

@torch.no_grad()
def RandomFlip(x, flip_num):
    X = []
    X.append(x)
    X.append(flip_inner(x, 1, 1))
    X.append(flip_inner(x, 0, 1))
    X.append(flip_inner(x, 1, 0))
    return torch.cat([X[i] for i in range(flip_num)], dim=0)

@torch.no_grad()
def GlobalRotation(x):
    return torch.cat((x, torch.rot90(x, 2, (2, 3)), torch.rot90(x, 1, (2, 3)), torch.rot90(x, 3, (2, 3))), dim=0)
