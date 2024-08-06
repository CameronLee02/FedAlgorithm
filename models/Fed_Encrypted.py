import copy
import torch

'''def FedAvg(w):
    w_avg = copy.deepcopy(w[0])
    for k in w_avg.keys():
        for i in range(1, len(w)):
            w_avg[k] += w[i][k]
        w_avg[k] = torch.div(w_avg[k], len(w))
    return w_avg'''


def FedAvg(encrypted_weights_list):
    """Perform federated averaging on encrypted weights."""
    avg_encrypted_weights = copy.deepcopy(encrypted_weights_list[0])
    for k in avg_encrypted_weights.keys():
        for i in range(1, len(encrypted_weights_list)):
            avg_encrypted_weights[k] = avg_encrypted_weights[k] + encrypted_weights_list[i][k]
    for k in avg_encrypted_weights.keys():
        avg_encrypted_weights[k] = avg_encrypted_weights[k] / len(encrypted_weights_list)
    return avg_encrypted_weights
