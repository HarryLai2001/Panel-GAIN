import numpy as np
import torch

def normalization(data):
    # Parameters
    _, dim = data.shape
    norm_data = np.zeros_like(data)

    # MixMax normalization
    min_val = np.zeros(dim)
    max_val = np.zeros(dim)

    # For each dimension
    for i in range(dim):
        min_val[i] = np.nanmin(data[:, i])
        max_val[i] = np.nanmax(data[:, i])
        norm_data[:, i] = (data[:, i] - min_val[i]) / (max_val[i] - min_val[i] + 1e-8)

    # Return norm_parameters for renormalization
    norm_parameters = {'min_val': min_val,
                       'max_val': max_val}

    return norm_data, norm_parameters


def renormalization(norm_data, norm_parameters):
    min_val = norm_parameters['min_val']
    max_val = norm_parameters['max_val']

    _, dim = norm_data.shape

    renorm_data = torch.zeros_like(norm_data)

    for i in range(dim):
        renorm_data[:, i] = norm_data[:, i] * (max_val[i] - min_val[i] + 1e-8) + min_val[i]

    return renorm_data