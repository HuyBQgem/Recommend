import os
import math
from time import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from utils import Tool



class CDAE(BaseRecommender):
    def __init__(self, dataset, model_conf, device):
        super(CDAE, self).__init__(dataset, model_conf)

        self.hidden_dim = model_conf['enc_dims']
        self.act = model_conf['act']
        self.corruption_ratio = model_conf['dropout']
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.device = device

        self.total_anneal_steps = model_conf['total_anneal_steps']
        self.anneal_cap = model_conf['anneal_cap']

        self.dropout = model_conf['dropout']

        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']

        self.lr = model_conf['lr']
        self.anneal = 0.
        self.update_count = 0

        self.user_embedding = nn.Embedding(self.num_users, self.hidden_dim)
        self.encoder = nn.Linear(self.num_items, self.hidden_dim)
        self.decoder = nn.Linear(self.hidden_dim, self.num_items)
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        self.to(self.device)


    def apply_activation(self, act_name, x):

        """
        Apply activation function
        :param act_name: name of the activation function
        :param x: input
        :return: output after activation
        """
        if act_name == 'sigmoid':
            return F.sigmoid(x)
        elif act_name == 'tanh':
            return F.tanh(x)
        elif act_name == 'relu':
            return F.relu(x)
        elif act_name == 'elu':
            return F.elu(x)
        else:
            raise NotImplementedError('Choose appropriate activation function. (current input: %s)' % act_name)



    def forward(self, user_id, rating_matrix):
        """
        Forward pass
        :param rating_matrix: rating matrix
        """
        # normalize the rating matrix
        user_degree = torch.norm(rating_matrix, 2, 1).view(-1, 1)  # user, 1
        item_degree = torch.norm(rating_matrix, 2, 0).view(1, -1)  # 1, item
        normalize = torch.sqrt(user_degree @ item_degree)
        zero_mask = normalize == 0
        normalize = torch.masked_fill(normalize, zero_mask.bool(), 1e-10)

        normalized_rating_matrix = rating_matrix / normalize

        # corrupt the rating matrix
        normalized_rating_matrix = F.dropout(rating_matrix, self.corruption_ratio, training=self.training)

        # build the collaborative denoising autoencoder
        #  
        enc = self.encoder(normalized_rating_matrix) +  self.user_embedding(user_id)
        enc = self.apply_activation(self.act, enc)
        dec = self.decoder(enc)
        
        return torch.sigmoid(dec)

    def train_model(self, dataset, evaluator, early_stop, logger, config):
        exp_config = config['Experiment']
        num_epochs = exp_config['num_epochs']
        print_step = exp_config['print_step']
        test_step = exp_config['test_step']
        test_from = exp_config['test_from']
        verbose = exp_config['verbose']
        log_dir = logger.log_dir

        # prepare dataset
        # dataset.set_eval_data('valid')
        users = np.arange(self.num_users)
        
        train_matrix = dataset.train_matrix.toarray()
        train_matrix = torch.FloatTensor(train_matrix)
        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        # for epoch
        start = time()
        for epoch in range(1, num_epochs + 1):
            self.train()

            epoch_loss = 0.0
            batch_loader = DataBatcher(users, batch_size=self.batch_size, drop_remain=False, shuffle=False)
            num_batches = len(batch_loader)
            # ======================== Train
            epoch_train_start = time()
            for b, batch_idx in enumerate(batch_loader):
                
                batch_matrix = train_matrix[batch_idx].to(self.device)
                batch_idx = torch.from_numpy(batch_idx).to(self.device)
                # print(type(batch_matrix),batch_matrix.is_cuda() , type(batch_idx), batch_idx.is_cuda())

                if self.total_anneal_steps > 0:
                    self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
                else:
                    self.anneal = self.anneal_cap

                batch_loss = self.train_model_per_batch(batch_idx, batch_matrix)
                epoch_loss += batch_loss

                if verbose and (b + 1) % verbose == 0:
                    print('batch %d / %d loss = %.4f' % (b + 1, num_batches, batch_loss))
            epoch_train_time = time() - epoch_train_start

            epoch_info = ['epoch=%3d' % epoch, 'loss=%.3f' % epoch_loss, 'train time=%.2f' % epoch_train_time]

            # ======================== Evaluate
            if (epoch >= test_from and epoch % test_step == 0) or epoch == num_epochs:
                self.eval()
                # evaluate model
                epoch_eval_start = time()

                test_score = evaluator.evaluate(self)
                test_score_str = ['%s=%.4f' % (k, test_score[k]) for k in test_score]

                updated, should_stop = early_stop.step(test_score, epoch)

                if should_stop:
                    logger.info('Early stop triggered.')
                    break
                else:
                    # save best parameters
                    if updated:
                        torch.save(self.state_dict(), os.path.join(log_dir, 'best_model.p'))
                        if self.anneal_cap == 1: 
                            print(self.anneal)

                epoch_eval_time = time() - epoch_eval_start
                epoch_time = epoch_train_time + epoch_eval_time

                epoch_info += ['epoch time=%.2f (%.2f + %.2f)' % (epoch_time, epoch_train_time, epoch_eval_time)]
                epoch_info += test_score_str
            else:
                epoch_info += ['epoch time=%.2f (%.2f + 0.00)' % (epoch_train_time, epoch_train_time)]

            if epoch % print_step == 0:
                logger.info(', '.join(epoch_info))

        total_train_time = time() - start

        return early_stop.best_score, total_train_time


    
    # def train_model_per_batch(self, batch_idx, batch_matrix):
    #     # self.train()
    #     self.optimizer.zero_grad()

    #     pred_matrix = self.forward(batch_idx, batch_matrix)

    #     # cross_entropy
    #     loss = F.binary_cross_entropy(pred_matrix, batch_matrix, reduction='sum')
    #     # loss  = -(F.log_softmax(pred_matrix, 1) * batch_matrix).sum(1).mean()

    #     loss.backward()
    #     self.optimizer.step()
    #     self.update_count += 1
    #     return loss
    
    def train_model_per_batch(self, batch_idx, batch_matrix, batch_weight=None):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        pred_matrix = self.forward(batch_idx, batch_matrix)

        # loss        
        # ce_loss = -(F.log_softmax(output, 1) * batch_matrix).mean()
        if batch_weight is None:
            loss = F.binary_cross_entropy(pred_matrix, batch_matrix, reduction='sum')
            # loss = -(F.log_softmax(pred_matrix, 1) * batch_matrix).sum(1).mean()
        else:
            loss = (F.binary_cross_entropy(pred_matrix, batch_matrix, reduction='sum') * batch_weight.view(pred_matrix.shape[0], -1))

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        self.update_count += 1

        return loss

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
            user_ids = torch.from_numpy(user_ids).to(self.device)
            eval_output = self.forward(user_ids, eval_input).detach().cpu().numpy()
            
            if eval_items is not None:
                eval_output[np.logical_not(eval_items)]=float('-inf')
            else:
                eval_output[batch_eval_pos.nonzero()] = float('-inf')
        self.train()
        return eval_output

    def restore(self, log_dir):
        with open(os.path.join(log_dir, 'best_model.p'), 'rb') as f:
            state_dict = torch.load(f)
        self.load_state_dict(state_dict)

    def get_output(self, dataset):
        test_eval_pos, test_eval_target, _ = dataset.test_data()
        num_users = len(test_eval_target)
        num_items = test_eval_pos.shape[1]
        eval_users = np.arange(num_users)
        user_iterator = DataBatcher(eval_users, batch_size=1024)
        output = np.zeros((num_users, num_items))
        for batch_user_ids in user_iterator:
            batch_pred = self.predict(batch_user_ids, test_eval_pos)
            output[batch_user_ids] += batch_pred
        return output