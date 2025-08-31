"""
main Code for PU-ScRR
Author: Sayantan Saha and Atif Hassan
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics import Accuracy
from torchvision import models
import numpy as np


def freeze(model):
    for param in model.parameters():
        if param.requires_grad:
            param.requires_grad = False
    return model


class ResNet(pl.LightningModule):    
    def __init__(self, lr, momentum, pretrained, resnet_type="18"):
        super(ResNet, self).__init__()
        if pretrained:
            if resnet_type == "18":
                self.model = torch.nn.Sequential(*list(models.resnet18(weights=models.ResNet18_Weights.DEFAULT).children())[:-1])
            elif resnet_type == "34":
                self.model = torch.nn.Sequential(*list(models.resnet34(weights=models.ResNet34_Weights.DEFAULT).children())[:-1])
            elif resnet_type == "50":
                self.model = torch.nn.Sequential(*list(models.resnet50(weights=models.ResNet50_Weights.DEFAULT).children())[:-1])
        else:
            if resnet_type == "18":
                self.model = torch.nn.Sequential(*list(models.resnet18(pretrained=False).children())[:-1])
            elif resnet_type == "34":
                self.model = torch.nn.Sequential(*list(models.resnet34(pretrained=False).children())[:-1])
            elif resnet_type == "50":
                self.model = torch.nn.Sequential(*list(models.resnet50(pretrained=False).children())[:-1])
        # self.model = freeze(self.model)
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
        
        self.learning_rate, self.momentum = lr, momentum
        self.train_acc, self.val_acc = Accuracy(task="binary"), Accuracy(task="binary")
        self.corr_train_acc, self.curr_train_acc = 0, 0
        self.max_val_acc, self.curr_val_acc = 0, 0
        self.test_preds = list()
        self.criterion = torch.nn.CrossEntropyLoss()
    
    def configure_optimizers(self):
        return torch.optim.SGD(self.parameters(), lr=self.learning_rate, momentum=self.momentum)

    def forward(self, x):
        return self.model(x)
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        preds = self.forward(x.float())
        loss = self.criterion(preds, y.long())
        self.train_acc.update(preds.argmax(1), y.long())
        self.log('train_acc', self.train_acc, on_epoch=True, prog_bar=True)
        return loss
    
    def training_epoch_end(self, outputs):
        self.curr_train_acc = self.train_acc.compute()
        if self.curr_val_acc == self.max_val_acc:
            self.corr_train_acc = self.curr_train_acc.item()
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x.float())
        self.val_acc.update(preds.argmax(1), y.long())
        self.log('val_acc', self.val_acc, on_epoch=True, prog_bar=True)
    
    def validation_epoch_end(self, outputs):
        self.curr_val_acc = self.val_acc.compute()
        if self.max_val_acc < self.curr_val_acc and self.current_epoch > 1:
            self.max_val_acc = self.curr_val_acc.item()
    
    def test_step(self, batch, batch_idx):
        try:
            x, y, _ = batch
        except:
            x, y = batch
        preds = self.forward(x.float())[:,0]
        self.test_preds.extend(preds.cpu().detach().numpy())