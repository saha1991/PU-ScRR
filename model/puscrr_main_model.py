"""
main Code for PU-ScRR
Author: Sayantan Saha and Atif Hassan
"""


from typing import Any, List, Optional, Union
from pytorch_lightning.utilities.types import EPOCH_OUTPUT, STEP_OUTPUT
import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchmetrics import F1Score
from torchvision import models
import numpy as np


def freeze(model):
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad = False
    return model


class ResNet(pl.LightningModule):    
    def __init__(self, gamma, lr, momentum, pretrained, network=None):
        super(ResNet, self).__init__()
        if network is None:
            if pretrained:
                self.model = torch.nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-1])
                self.model2 = torch.nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-1])
            else:
                self.model = torch.nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1])
                self.model2 = torch.nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1])
            # self.model = freeze(self.model)
            # self.model2 = freeze(self.model2)
            self.model.append(torch.nn.Flatten(start_dim=1))
            self.model.append(torch.nn.ReLU())
            self.model.append(nn.Linear(512, 1024))
            self.model.append(torch.nn.ReLU())
            self.model.append(nn.Linear(1024, 512))
            self.model.append(torch.nn.ReLU())
            self.model.append(nn.Linear(512, 1024))
            self.model.append(torch.nn.ReLU())
            self.model.append(nn.Linear(1024, 256))
            self.model.append(torch.nn.ReLU())
            self.model.append(nn.Linear(256, 2))
        else:
            self.model = network
        
        self.linear = nn.Linear(512, 1)
                
        self.gamma = gamma
        self.learning_rate, self.momentum = lr, momentum
        self.train_acc, self.val_acc, self.f1_score = Accuracy(task="binary"), Accuracy(task="binary"), F1Score(task='binary')
        self.corr_train_acc, self.curr_train_acc = 0, 0
        # self.max_f1_score, self.curr_f1_score = 0, 0
        self.max_val_acc, self.curr_val_acc = 0, 0
        # self.corr_val_acc, self.curr_val_acc = 0, 0
        self.test_preds, self.all_alpha_values, self.all_true_labels = list(), list(), list()
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def forward(self, x):
        out = torch.nn.Softmax()(self.model(x))
        
        alphas = torch.squeeze(torch.nn.ReLU()(self.model2(x)))
        alphas = self.linear(alphas)
        return out, torch.squeeze(alphas)
    
    def custom_bce_loss(self, preds, y, alphas, gamma):
        a, b = torch.clamp(torch.log(preds[:,1]), min=-100), torch.clamp(torch.log(preds[:,0]), min=-100)
        return -1 * (torch.mean(y * a + torch.nn.ReLU()(alphas) * (1-y) * b)) + gamma * torch.sum(torch.abs(1-alphas))
    
    def training_step(self, batch, batch_idx):
        x, y, true_labels = batch
        preds, alphas = self.forward(x.float())
        gamma = self.gamma
        loss = self.custom_bce_loss(preds, y.float(), alphas, gamma)
        self.train_acc.update(preds.argmax(1), y.long())
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)
        return loss
    
    def training_epoch_end(self, outputs):
        self.curr_train_acc = self.train_acc.compute()
        if self.curr_val_acc == self.max_val_acc:
            self.corr_train_acc = self.curr_train_acc.item()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds, _ = self.forward(x.float())
        # self.f1_score.update(preds.argmax(1), y.long())
        self.val_acc.update(preds.argmax(1), y.long())
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
        # self.log('f1_score', self.f1_score, on_epoch=True, prog_bar=True)
    
    def validation_epoch_end(self, outputs):
        # self.curr_f1_score = self.f1_score.compute()
        self.curr_val_acc = self.val_acc.compute()
        # if self.max_f1_score < self.curr_f1_score and self.current_epoch > 1:
        #     self.max_f1_score = self.curr_f1_score.item()
        #     self.corr_val_acc = self.curr_val_acc.item()
        if self.max_val_acc < self.curr_val_acc and self.current_epoch > 1:
            self.max_val_acc = self.curr_val_acc.item()
    
    # def test_step(self, batch, batch_idx):
    #     try:
    #         x, y, true_labels = batch
    #     except:
    #         x, y = batch
    #     preds, alpha = self.forward(x.float())
    #     self.test_preds.extend(np.argmax(preds.cpu().detach().numpy(), axis=1))
    #     self.all_alpha_values.extend(alpha.cpu().detach().numpy())
    #     self.all_true_labels.extend(true_labels.cpu().detach().numpy())