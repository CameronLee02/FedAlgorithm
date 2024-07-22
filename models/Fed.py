import copy
import torch

'''def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg'''


def FedAvg(weight_list):
    num_contributions = len(weight_list)
    w_avg = copy.deepcopy(weight_list[0])
    for k in w_avg.keys():
        for i in range(1, num_contributions):
            w_avg[k] += weight_list[i][k]
        w_avg[k] = torch.div(w_avg[k], num_contributions)
    return w_avg
