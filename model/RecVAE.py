import numpy as np
from copy import deepcopy
import os
from time import time
import torch
from torch import nn
from torch.nn import functional as F
from base.BaseRecommender import BaseRecommender
from dataloader.DataBatcher import DataBatcher
from utils import Tool


def swish(x):
    return x.mul(torch.sigmoid(x))

def log_norm_pdf(x, mu, logvar):
    return -0.5*(logvar + np.log(2 * np.pi) + (x - mu).pow(2) / logvar.exp())


class CompositePrior(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, mixture_weights=[3/20, 3/4, 1/10]):
        super(CompositePrior, self).__init__()
        
        self.mixture_weights = mixture_weights
        
        self.mu_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.mu_prior.data.fill_(0)
        
        self.logvar_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_prior.data.fill_(0)
        
        self.logvar_uniform_prior = nn.Parameter(torch.Tensor(1, latent_dim), requires_grad=False)
        self.logvar_uniform_prior.data.fill_(10)
        
        self.encoder_old = Encoder(hidden_dim, latent_dim, input_dim)
        self.encoder_old.requires_grad_(False)
        
    def forward(self, x, z):
        post_mu, post_logvar = self.encoder_old(x, 0)
        
        stnd_prior = log_norm_pdf(z, self.mu_prior, self.logvar_prior)
        post_prior = log_norm_pdf(z, post_mu, post_logvar)
        unif_prior = log_norm_pdf(z, self.mu_prior, self.logvar_uniform_prior)
        
        gaussians = [stnd_prior, post_prior, unif_prior]
        gaussians = [g.add(np.log(w)) for g, w in zip(gaussians, self.mixture_weights)]
        
        density_per_gaussian = torch.stack(gaussians, dim=-1)
                
        return torch.logsumexp(density_per_gaussian, dim=-1)

    
class Encoder(nn.Module):
    def __init__(self, hidden_dim, latent_dim, input_dim, eps=1e-1):
        super(Encoder, self).__init__()
         
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.ln1 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.ln2 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.ln3 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.ln4 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc5 = nn.Linear(hidden_dim, hidden_dim)
        self.ln5 = nn.LayerNorm(hidden_dim, eps=eps)
        self.fc_mu = nn.Linear(hidden_dim, latent_dim)
        self.fc_logvar = nn.Linear(hidden_dim, latent_dim)
        
    def forward(self, x, dropout_rate):
        norm = x.pow(2).sum(dim=-1).sqrt()
        x = x / norm[:, None]
    
        x = F.dropout(x, p=dropout_rate, training=self.training)
        
        h1 = self.ln1(swish(self.fc1(x)))
        h2 = self.ln2(swish(self.fc2(h1) + h1))
        h3 = self.ln3(swish(self.fc3(h2) + h1 + h2))
        h4 = self.ln4(swish(self.fc4(h3) + h1 + h2 + h3))
        h5 = self.ln5(swish(self.fc5(h4) + h1 + h2 + h3 + h4))
        return self.fc_mu(h5), self.fc_logvar(h5)
    

class RecVAE(BaseRecommender, nn.Module):
    def __init__(self, dataset, model_conf, device):
        super(RecVAE, self).__init__(dataset, model_conf)
        self.dataset = dataset

        self.hidden_dim = model_conf['enc_dims']
        self.num_users = dataset.num_users
        self.num_items = dataset.num_items
        self.latent_dim = model_conf["latent_dim"]
        self.dropout = model_conf['dropout']
        self.gamma = model_conf["gamma"]

        # self.calculate_loss = model_conf["calculate_loss"]
        self.input_dim = dataset.train_matrix.shape[1]

        self.device = device
        self.total_anneal_steps = model_conf['total_anneal_steps']
        self.anneal_cap = model_conf['anneal_cap']
        self.update_count = 0
        
        self.batch_size = model_conf['batch_size']
        self.test_batch_size = model_conf['test_batch_size']
        self.lr = model_conf['lr']

        self.encoder = Encoder(self.hidden_dim, self.latent_dim, self.input_dim)
        self.prior = CompositePrior(self.hidden_dim, self.latent_dim, self.input_dim)
        self.decoder = nn.Linear(self.latent_dim, self.input_dim)

        # self.encoder = Encoder(hidden_dim, latent_dim, input_dim)
        # self.prior = CompositePrior(hidden_dim, latent_dim, input_dim)
        # self.decoder = nn.Linear(latent_dim, input_dim)

        self.optimizer = torch.optim.Adam(self.parameters(), self.lr)
        self.to(self.device)
        
    def reparameterize(self, mu, logvar):
        if self.training:
            # logvar_ten = torch.FloatTensor(logvar)
            std = torch.exp(0.5*logvar)
            # std = torch.Tensor(std)
            eps = torch.randn_like(std)
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, user_ratings):
        mu, logvar = self.encoder(user_ratings, self.dropout)   
        z = self.reparameterize(mu, logvar)
        x_pred = self.decoder(z)
        return x_pred
        
    def train_model_per_batch(self, batch_matrix, batch_weight=None):
        # zero grad
        self.optimizer.zero_grad()

        # model forwrad
        output = self.forward(batch_matrix)

        # loss        
        # ce_loss = -(F.log_softmax(output, 1) * batch_matrix).mean()
        if batch_weight is None:
            ce_loss = -(F.log_softmax(output, 1) * batch_matrix).sum(1).mean()
        else:
            ce_loss = -((F.log_softmax(output, 1) * batch_matrix) * batch_weight.view(output.shape[0], -1)).sum(1).mean()

        # loss = ce_loss + kl_loss * self.anneal
        loss = ce_loss

        # backward
        loss.backward()

        # step
        self.optimizer.step()

        self.update_count += 1

        return loss
    
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

                if self.total_anneal_steps > 0:
                    self.anneal = min(self.anneal_cap, 1. * self.update_count / self.total_anneal_steps)
                else:
                    self.anneal = self.anneal_cap

                batch_loss = self.train_model_per_batch(batch_matrix)
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
                        if self.anneal_cap == 1: print(self.anneal)

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

    def predict(self, user_ids, eval_pos_matrix, eval_items=None):
        self.eval()
        batch_eval_pos = eval_pos_matrix[user_ids]
        with torch.no_grad():
            eval_input = torch.Tensor(batch_eval_pos.toarray()).to(self.device)
            eval_output = self.forward(eval_input).detach().cpu().numpy()
            
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

    def user_embedding(self, input_matrix):
        with torch.no_grad():
            user_embedding = torch.zeros(self.num_users, self.enc_dims[-1])
            users = np.arange(self.num_users)

            input_matrix = torch.FloatTensor(input_matrix.toarray())

            batch_size = self.test_batch_size
            batch_loader = DataBatcher(users, batch_size=batch_size, drop_remain=False, shuffle=False)
            for b, (batch_user_idx) in enumerate(batch_loader):
                batch_matrix = input_matrix[batch_user_idx]
                batch_matrix = torch.Tensor(batch_matrix).to(self.device)

                h = F.dropout(F.normalize(batch_matrix), p=self.dropout, training=self.training)
                for layer in self.encoder:
                    h = layer(h)
                batch_emb = h[:, :self.enc_dims[-1]]  # mu

                user_embedding[batch_user_idx] += batch_emb.detach().cpu()

        return user_embedding.detach().cpu().numpy()

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
