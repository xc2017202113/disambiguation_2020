import torch

def euclidean_distance(x,y):

    return torch.sqrt(torch.max(torch.sum(torch.pow(x - y,2), dim=1,keepdim=True), torch.tensor(1e-12)))


def triplet_loss(pos_dis, neg_dis):
    margin = torch.tensor(1.0).float()
    return torch.mean(torch.max(torch.tensor(0.0).float(), (torch.pow(pos_dis,2) - torch.pow(neg_dis,2)) + margin))
    #return torch.mean(pos_dis-neg_dis)

def dis_acc(pos_dis,neg_dis):
    return (neg_dis-torch.tensor(2.0).float()>pos_dis).float()