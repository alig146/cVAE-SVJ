##################################################################################
# VAE on DarkMachine dataset with 3D Sparse Loss                                 # 
# Author: B. Orzani (Universidade Estadual Paulista, Brazil), M. Pierini (CERN)  #
##################################################################################

import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets
from sklearn.utils import shuffle
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KernelDensity
from pickle import dump
import numpy as np
import h5py
from tqdm import tqdm

from data_utils import save_npy, save_csv, read_npy, save_run_history, quick_logit, logit_transform_inverse
from network_utils import train_convnet, test_convnet
import VAE_NF_Conv2D as VAE


class ConvNetRunner:
    def __init__(self, args):

        # Hyperparameters
        self.data_save_path = args.data_save_path
        self.model_save_path = args.model_save_path
        self.model_name = args.model_name
        self.num_epochs = args.num_epochs
        self.epoch1 = args.epoch1
        self.epoch2 = args.epoch2
        self.num_gen_SR = args.num_gen_SR
        self.num_classes = args.num_classes
        self.batch_size = args.batch_size
        self.test_batch_size = args.test_batch_size
        self.lr_default = args.learning_rate
        self.learning_rate = args.learning_rate
        self.latent_dim = args.latent_dim
        self.h_size = args.made_h_size
        self.beta = args.beta
        self.test_data_save_path = args.test_data_save_path

        self.network = args.network
        self.flow = args.flow 

        if self.flow == 'noflow':
            self.model = VAE.ConvNet(args)
            self.flow_ID = 'NoF'
        elif self.flow == 'planar':
            self.model = VAE.PlanarVAE(args)
            self.flow_ID = 'Planar'
        elif self.flow == 'orthosnf':
            self.model = VAE.OrthogonalSylvesterVAE(args)
            self.flow_ID = 'Ortho'
        elif self.flow == 'householdersnf':
            self.model = VAE.HouseholderSylvesterVAE(args)
            self.flow_ID = 'House'
        elif self.flow == 'triangularsnf':
            self.model = VAE.TriangularSylvesterVAE(args)
            self.flow_ID = 'Tri'
        elif self.flow == 'iaf':
            self.model = VAE.IAFVAE(args)
            self.flow_ID = 'IAF'
        elif self.flow == 'convflow':
            self.model = VAE.ConvFlowVAE(args)
            self.flow_ID = 'ConvF'
        elif self.flow == 'maf':
            self.model = VAE.MAF_VAE(args)
            self.flow_ID = 'MAF'
        else:
            raise ValueError('Invalid flow choice')
        
        self.model_name = self.model_name%self.flow_ID
        self.model = self.model.cuda()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)

        self.preprocess_data()

    def preprocess_data(self):

        nFeat = 6

        outerdata_train = np.load("Datasets/preprocessed_data_6var/outerdata_train_6var.npy")
        outerdata_test = np.load("Datasets/preprocessed_data_6var/outerdata_test_6var.npy")

        outerdata_train = outerdata_train[outerdata_train[:,nFeat+1]==0]
        outerdata_test = outerdata_test[outerdata_test[:,nFeat+1]==0]

        data_train = outerdata_train[:,1:nFeat+1]
        print('shape of data_train: ', data_train.shape)
        data_test = outerdata_test[:,1:nFeat+1]
        print('shape of data_test: ', data_test.shape)

        data = np.concatenate((data_train, data_test), axis=0)
        print('shape of data: ', data.shape)

        cond_data_train = outerdata_train[:,0]
        print('shape of cond_train', cond_data_train.shape)
        cond_data_test = outerdata_test[:,0]
        print('shape of cond_test', cond_data_test.shape)

        cond_data = np.concatenate((cond_data_train, cond_data_test), axis=0)
        print('shape of data: ', cond_data.shape)


        # scalar_x = StandardScaler()
        # scalar_x.fit(data)
        # data = scalar_x.transform(data)
        # self.scalar_x = scalar_x

        # scalar_cond = StandardScaler()
        # cond_data = np.reshape(cond_data, [-1, 1])
        # scalar_cond.fit(cond_data)
        # cond_data = scalar_cond.transform(cond_data)
        # self.scalar_cond = scalar_cond

        x_max = np.empty(nFeat)
        for i in range(0,data.shape[1]):
            x_max[i] = np.max(np.abs(data[:,i]))
            if np.abs(x_max[i]) > 0: 
                data[:,i] = data[:,i]/x_max[i]
            else:
                pass

        self.data = data
        self.x_max = x_max

        cond_max = np.max(np.abs(cond_data))
        if np.abs(cond_max) > 0:
            cond_data = cond_data/cond_max
        else:
            pass

        self.cond_data = cond_data
        self.cond_max = cond_max
        self.N_bkg_SB = cond_data.shape[0]


        trainsize = outerdata_train.shape[0]
        self.trainsize = trainsize
        
        x_train = data[:trainsize]
        x_test = data[trainsize:]
        y_train = cond_data[:trainsize]
        y_test = cond_data[trainsize:]

        image_size = x_train.shape[1]
        original_dim = image_size
        x_train = np.reshape(x_train, [-1, original_dim])
        x_test = np.reshape(x_test, [-1, original_dim])
        
        x_train = x_train.astype('float32')
        x_test = x_test.astype('float32')
        
        y_train = np.reshape(y_train, [-1, 1])
        y_test = np.reshape(y_test, [-1, 1])
        
        y_train = y_train.astype('float32')
        y_test = y_test.astype('float32')

        
        self.x_train = x_train
        self.met_train = y_train 
        
        # Val data
        self.x_val = x_test
        self.met_val = y_test
        

        ####################################
        # process inner data

        innerdata_train = np.load("Datasets/preprocessed_data_6var/innerdata_train_6var.npy")
        innerdata_train = innerdata_train[innerdata_train[:,nFeat+1]==0]
        y_innerdata_train = innerdata_train[:,0]
        self.y_innerdata_train = y_innerdata_train
        

    def trainer(self):
        self.train_loader = DataLoader(dataset = self.x_train, batch_size = self.batch_size, shuffle=True)
        self.metTr_loader = DataLoader(dataset = self.met_train, batch_size = self.batch_size, shuffle=True)

        self.val_loader = DataLoader(dataset = self.x_val, batch_size = self.batch_size, shuffle=False)
        self.metVa_loader = DataLoader(dataset = self.met_val, batch_size = self.batch_size, shuffle=False)

        # to store training history
        self.x_graph = []
        # self.train_y_rec = []
        self.train_y_kl = []
        self.train_y_loss = []
        
        # self.val_y_rec = []
        self.val_y_kl = []
        self.val_y_loss = []

        # print('Model Parameter: ', self.model)
        print('Model Type: %s'%self.flow_ID)
        print('Initiating training, validation processes ...')
        for epoch in range(self.num_epochs):
            self.x_graph.append(epoch)
            print('Starting to train ...')

            # adjust learning rate
            epoch1 = self.epoch1
            epoch2 = self.epoch2

            if epoch < epoch1*4:
                itr = epoch // epoch1
                self.learning_rate = self.lr_default/(2**itr)
                self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
            else:
                itr = 4 + (epoch-epoch1*4) // epoch2
                self.learning_rate = self.lr_default/(2**itr)
                self.optimizer = torch.optim.SGD(self.model.parameters(), lr=self.learning_rate, momentum=0.9, weight_decay=1e-6, nesterov=True)


            # training
            tr_loss_aux = 0.0
            tr_kl_aux = 0.0
            tr_rec_aux = 0.0
            self.train_z_mu = np.empty(0)
            self.train_z_var = np.empty(0)
            if self.flow_ID == 'IAF':
                self.train_h_context = np.empty(0)


            for y, (x_train, met_tr) in tqdm(enumerate(zip(self.train_loader, self.metTr_loader))):
                if y == (len(self.train_loader)): break

                if self.flow_ID == 'IAF': 
                    z_mu, z_var, h_context, tr_loss, tr_kl, self.model = train_convnet(self.model, x_train, met_tr, self.optimizer, batch_size=self.batch_size, beta=self.beta, flow_id = self.flow_ID)
                else:
                    z_mu, z_var, tr_loss, tr_kl, self.model = train_convnet(self.model, x_train, met_tr, self.optimizer, batch_size=self.batch_size, beta=self.beta, flow_id = self.flow_ID)
                
                tr_loss_aux += float(tr_loss)
                tr_kl_aux += float(tr_kl)
                
                if self.train_z_mu.shape[0] == 0:
                    self.train_z_mu = z_mu.cpu().detach().numpy()
                    self.train_z_var = z_var.cpu().detach().numpy()
                    if self.flow_ID == 'IAF':
                        self.train_h_context = h_context.cpu().detach().numpy()
                else:
                    self.train_z_mu = np.concatenate((self.train_z_mu, z_mu.cpu().detach().numpy()))
                    self.train_z_var = np.concatenate((self.train_z_var, z_var.cpu().detach().numpy()))
                    if self.flow_ID == 'IAF':
                        self.train_h_context = np.concatenate((self.train_h_context, h_context.cpu().detach().numpy()))

            print('Moving to validation stage ...')
            # validation
            val_loss_aux = 0.0
            val_kl_aux = 0.0
            val_rec_aux = 0.0

            for y, (x_val, met_va) in tqdm(enumerate(zip(self.val_loader, self.metVa_loader))):
                if y == (len(self.val_loader)): break
                
                #Test
                _, val_loss, val_kl = test_convnet(self.model, x_val, met_va, batch_size=self.batch_size, beta=self.beta, flow_id = self.flow_ID)

                val_loss_aux += float(val_loss)
                val_kl_aux += float(val_kl)

            self.train_y_loss.append(tr_loss_aux/(len(self.train_loader)))
            self.train_y_kl.append(tr_kl_aux/(len(self.train_loader)))
               
            self.val_y_loss.append(val_loss_aux/(len(self.val_loader)))
            self.val_y_kl.append(val_kl_aux/(len(self.val_loader)))
                
            print('Epoch: {} -- Train loss: {}  -- Val loss: {}'.format(epoch, 
                                                                         tr_loss_aux/(len(self.train_loader)), 
                                                                         val_loss_aux/(len(self.val_loader))))
            if (epoch == 0):
                self.best_val_loss = val_loss_aux/(len(self.val_loader))
                self.best_model = self.model
                
                self.best_train_z_mu = self.train_z_mu
                self.best_train_z_var = self.train_z_var
                if self.flow_ID == 'IAF':
                    self.best_train_h_context = self.train_h_context
            if (val_loss_aux/(len(self.val_loader))<self.best_val_loss):
                self.best_model = self.model
                self.best_val_loss = val_loss_aux/(len(self.val_loader))
                
                self.best_train_z_mu = self.train_z_mu
                self.best_train_z_var = self.train_z_var
                if self.flow_ID == 'IAF':
                    self.best_train_h_context = self.train_h_context
                print('Best Model Yet')


        print("Save latent info.")
        save_npy(np.array(self.best_train_z_mu), self.model_save_path + 'best_latent_mean_6var_%s.npy' %self.model_name)
        save_npy(np.array(self.best_train_z_var), self.model_save_path + 'best_latent_std_6var_%s.npy' %self.model_name)
        if self.flow_ID == 'IAF':
            save_npy(np.array(self.best_train_h_context), self.model_save_path + 'best_h_context_6var_%s.npy' %self.model_name)
        
        save_run_history(self.best_model, self.model, self.model_save_path, self.model_name, 
                            self.x_graph, self.train_y_kl, self.train_y_loss, hist_name='TrainHistory')
        
        save_npy(np.array(self.train_y_loss), self.data_save_path + '%s_train_loss.npy' %self.model_name)
        save_npy(np.array(self.train_y_kl), self.data_save_path + '%s_train_kl.npy' %self.model_name)
        save_npy(np.array(self.val_y_loss), self.data_save_path + '%s_val_loss.npy' %self.model_name)
        save_npy(np.array(self.val_y_kl), self.data_save_path + '%s_val_kl.npy' %self.model_name)
        

        print('Network Run Complete')


    def event_generater_SB(self):
        self.model.load_state_dict(torch.load(self.model_save_path + 'BEST_%s.pt' %self.model_name, map_location=torch.device('cpu')))
        self.model.eval()
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            best_z_mu = np.load(self.model_save_path + 'best_latent_mean_6var_%s.npy' %self.model_name, allow_pickle=True)
            best_z_logvar = np.load(self.model_save_path + 'best_latent_std_6var_%s.npy' %self.model_name, allow_pickle=True)
            if self.flow_ID == 'IAF':
                best_h_context = np.load(self.model_save_path + 'best_h_context_6var_%s.npy' %self.model_name, allow_pickle=True)

            best_z_var = np.exp(best_z_logvar)
            best_z_std = np.sqrt(best_z_var)

            z_samples = np.empty([self.N_bkg_SB, self.latent_dim])
            if self.flow_ID == 'IAF':
                h_samples = np.empty([self.N_bkg_SB, self.h_size])

            l=0
            for i in range(0,self.N_bkg_SB):
                if self.flow_ID == 'IAF':
                    h_samples[i,:] = best_h_context[i%self.trainsize,:]
                for j in range(0,self.latent_dim):
                    z_samples[l,j] = np.random.normal(best_z_mu[i%self.trainsize,j], 0.05+best_z_std[i%self.trainsize,j])
                    # z_samples[l,j] = np.random.normal(0,1)
                l=l+1
                
            z_samples_tensor = torch.from_numpy(z_samples.astype('float32')).to(device)
            if self.flow_ID == 'IAF':
                h_samples_tensor = torch.from_numpy(h_samples.astype('float32')).to(device)
            cond_data_tensor = torch.from_numpy(np.reshape(self.cond_data, [-1, 1]).astype('float32')).to(device)

            if self.flow_ID == 'ConvF' or self.flow_ID == 'MAF':
                z_samples_tensor, _ = self.model.flow(z_samples_tensor)
            if self.flow_ID == 'IAF':
                z_samples_tensor, _ = self.model.flow(z_samples_tensor, h_samples_tensor)

            new_events = self.model.decode(z_samples_tensor, cond_data_tensor).data.cpu().numpy()

            for i in range(0,new_events.shape[1]):
                new_events[:,i]=new_events[:,i]*self.x_max[i]
            # new_events = self.scalar_x.inverse_transform(new_events)

            np.savetxt(self.data_save_path + 'LHCO2020_cB-VAE_events_%s_SB.csv' %self.model_name, new_events)

        print("Done generating SB events.")


    def event_generater_SR(self):
        self.model.load_state_dict(torch.load(self.model_save_path + 'BEST_%s.pt' %self.model_name, map_location=torch.device('cpu')))
        self.model.eval()
        with torch.no_grad():
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

            # fit and generate mjj values
            KDE_bandwidth = 0.01
            mjj_logit = quick_logit(self.y_innerdata_train)
            train_mjj_vals = logit_transform_inverse(KernelDensity(
                bandwidth=KDE_bandwidth, kernel='gaussian').fit(
                    mjj_logit.reshape(-1, 1)).sample(self.num_gen_SR),
                                                        max(self.y_innerdata_train).item(),
                                                        min(self.y_innerdata_train).item())

            if np.abs(self.cond_max) > 0:
                train_mjj_vals_scaled = train_mjj_vals/self.cond_max
            else:
                train_mjj_vals_scaled = train_mjj_vals

        
            best_z_mu = np.load(self.model_save_path + 'best_latent_mean_6var_%s.npy' %self.model_name, allow_pickle=True)
            best_z_logvar = np.load(self.model_save_path + 'best_latent_std_6var_%s.npy' %self.model_name, allow_pickle=True)
            if self.flow_ID == 'IAF':
                best_h_context = np.load(self.model_save_path + 'best_h_context_6var_%s.npy' %self.model_name, allow_pickle=True)

            best_z_var = np.exp(best_z_logvar)
            best_z_std = np.sqrt(best_z_var)

            z_samples = np.empty([self.num_gen_SR, self.latent_dim])
            if self.flow_ID == 'IAF':
                h_samples = np.empty([self.num_gen_SR, self.h_size])

            l=0
            for i in range(0,self.num_gen_SR):
                if self.flow_ID == 'IAF':
                    h_samples[i,:] = best_h_context[i%self.trainsize,:]
                for j in range(0,self.latent_dim):
                    z_samples[l,j] = np.random.normal(best_z_mu[i%self.trainsize,j], 0.05+best_z_std[i%self.trainsize,j])
                    # z_samples[l,j] = np.random.normal(0,1)
                l=l+1
                
            z_samples_tensor = torch.from_numpy(z_samples.astype('float32')).to(device)
            if self.flow_ID == 'IAF':
                h_samples_tensor = torch.from_numpy(h_samples.astype('float32')).to(device)
            cond_data_tensor = torch.from_numpy(np.reshape(train_mjj_vals_scaled, [-1, 1]).astype('float32')).to(device)

            if self.flow_ID == 'ConvF' or self.flow_ID == 'MAF':
                z_samples_tensor, _ = self.model.flow(z_samples_tensor)
            if self.flow_ID == 'IAF':
                z_samples_tensor, _ = self.model.flow(z_samples_tensor, h_samples_tensor)

            new_events = self.model.decode(z_samples_tensor, cond_data_tensor).data.cpu().numpy()

            for i in range(0,new_events.shape[1]):
                new_events[:,i]=new_events[:,i]*self.x_max[i]
            # new_events = self.scalar_x.inverse_transform(new_events)

            # np.savetxt('/workdir/huichi/NF-C-VAE/data_save/LHCO2020_cB-VAE_NF_events_6var_SR.csv', new_events)
            np.savetxt(self.data_save_path + 'LHCO2020_cB-VAE_events_%s_SR.csv' %self.model_name, new_events)

        print("Done generating SR events.")
    






        

