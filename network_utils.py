from ast import arg
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
import numpy as np


#Sparse loss function    
def compute_loss(x, x_decoded, mean, logvar, log_det_j, batch_size=1, beta=1, flow_id='iaf'):
    mse = nn.MSELoss()
    KL_divergence = 0.5 * torch.sum((torch.pow(mean, 2) + torch.exp(logvar) - logvar - 1.0)).sum() / batch_size 
    loss = torch.mean((1-beta)*mse(x, x_decoded) + (beta * KL_divergence))
    return loss, (beta * KL_divergence)

# Train
def train_convnet(model, x_train, met_train, optimizer, batch_size, beta, flow_id):
    input_train = x_train.cuda().float()
    met_train = met_train.cuda().float()
    # wt_train = wt_train[:].cuda()
    model.train()   

    if flow_id == 'IAF':
        x_decoded, z_mu, z_var, log_det_j, z0, h_context = model(input_train, met_train)
    else:
        x_decoded, z_mu, z_var, log_det_j, z0, zk = model(input_train, met_train)

    tr_loss, tr_kl = compute_loss(input_train, x_decoded, z_mu, z_var, log_det_j, batch_size=batch_size, beta=beta, flow_id=flow_id)
    
    # Backprop and perform Adam optimisation
    optimizer.zero_grad()
    tr_loss.backward()
    optimizer.step()

    if flow_id == 'IAF':
        return z_mu, z_var, h_context, tr_loss, tr_kl, model
    else:
        return z_mu, z_var, tr_loss, tr_kl, model

# Test/Validate
def test_convnet(model, x_test, met_test, batch_size, beta, flow_id):
    model.eval()
    with torch.no_grad():
        input_test = x_test.cuda().float()
        met_test = met_test.cuda().float()

        x_decoded, z_mu, z_var, log_det_j, z0, zk = model(input_test, met_test)
        
        te_loss, te_kl = compute_loss(input_test, x_decoded, z_mu, z_var, log_det_j, batch_size=batch_size, beta=beta, flow_id=flow_id)

    return x_decoded, te_loss, te_kl

