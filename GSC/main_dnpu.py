import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import os
import sklearn
import scipy
from sklearn import model_selection

from torch.utils.data import DataLoader, Dataset, random_split

from sklearn.preprocessing import LabelEncoder

import numpy as np

from tqdm import tqdm
from itertools import chain
from torchvision import transforms

import pytorch_lightning as L
import torchmetrics

import time

from pytorch_lightning.loggers import TensorBoardLogger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

import torch
import torch.nn as nn

class M3(nn.Module):
    def __init__(self, n_output, n_channel = 64):
        super().__init__()
        self.layer_norm = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        # self.conv4 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3)
        # self.bn4 = nn.BatchNorm1d(4 * n_channel)
        # self.pool4 = nn.MaxPool1d(4)

        # self.conv_ext = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3)
        # self.bn_ext = nn.BatchNorm1d(4 * n_channel)
        # self.pool_ext = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(2 * n_channel, n_output)
        # self.fc2 = nn.Linear(n_channel, n_output)

    def forward(self, x):
        x = self.layer_norm(x[:,:,::4])
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.tanh(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.tanh(self.bn3(x))
        x = self.pool3(x)
        # x = self.conv4(x)
        # x = F.tanh(self.bn4(x))
        # x = self.pool4(x)

        # x = self.conv_ext(x)
        # x = F.tanh(self.bn_ext(x))
        # x = self.pool_ext(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        # x = F.tanh(self.fc1(x))
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class M4(nn.Module):
    def __init__(self, n_output, n_channel = 64):
        super().__init__()
        self.layer_norm = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        # self.pool4 = nn.MaxPool1d(4)

        self.conv_ext = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn_ext = nn.BatchNorm1d(2 * n_channel)
        # self.pool_ext = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(2 * n_channel, n_output)
        # self.fc2 = nn.Linear(n_channel, n_output)

    def forward(self, x):
        x = self.layer_norm(x[:,:,::4])
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        # x = self.pool4(x)

        x = self.conv_ext(x)
        x = F.tanh(self.bn_ext(x))
        # x = self.pool_ext(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        # x = F.tanh(self.fc1(x))
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class LitClassifier(L.LightningModule):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.model = kwargs.get('model')
        self.num_classes = kwargs.get('num_classes')
        self.max_lr = kwargs.get('max_lr')

        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.valid_acc = torchmetrics.Accuracy(task='multiclass', num_classes=self.num_classes)
        self.best_acc  = torchmetrics.MaxMetric()

    def forward(self, x):
        return self.model(x)
    
    def _shared_forward_step(self, batch):
        features, true_labels = batch
        logits = self(features).squeeze()
        loss = F.cross_entropy(logits, true_labels.long())
        predicted_labels = torch.argmax(logits, dim=1)

        return loss, true_labels, predicted_labels

    def training_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_forward_step(batch)
        self.log("train_loss", loss)

        self.model.eval()
        with torch.no_grad():
            _, true_labels, predicted_labels = self._shared_forward_step(batch)
        self.train_acc.update(predicted_labels, true_labels)
        self.log("train_acc", self.train_acc, on_epoch=True, on_step=False)
        self.model.train()

        return loss

    def validation_step(self, batch, batch_idx):
        loss, true_labels, predicted_labels = self._shared_forward_step(batch)
        self.log("valid_loss", loss)
        self.valid_acc(predicted_labels, true_labels)
        self.log("valid_acc", self.valid_acc, on_epoch = True, on_step=False, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), weight_decay=1e-4)

        lr_scheduler_config = {
            # REQUIRED: The scheduler instance
            "scheduler": torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=self.max_lr, total_steps=self.trainer.estimated_stepping_batches),
            # The unit of the scheduler's step size, could also be 'step'.
            # 'epoch' updates the scheduler on epoch end whereas 'step'
            # updates it after a optimizer update.
            "interval": "epoch",
            # How many epochs/steps should pass between calls to
            # `scheduler.step()`. 1 corresponds to updating the learning
            # rate after every epoch/step.
            "frequency": 1,
            # Metric to to monitor for schedulers like `ReduceLROnPlateau`
            "monitor": "val_loss",
            # If set to `True`, will enforce that the value specified 'monitor'
            # is available when the scheduler is updated, thus stopping
            # training if not found. If set to `False`, it will only produce a warning
            "strict": True,
            # If using the `LearningRateMonitor` callback to monitor the
            # learning rate progress, this keyword can be used to specify
            # a custom logged name
            "name": None,
        }

        return [optimizer], [lr_scheduler_config]
    
class LitDataModule(L.LightningDataModule):
    def __init__(self, num_classes, batch_size = 32, num_workers = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        self.trainset = torch.load("GSC/datasets/dnpu_measurements/trainset_kernel=8_12classes.pt", weights_only=False)
        self.testset = torch.load("GSC/datasets/dnpu_measurements/testset_kernel=8_12classes.pt", weights_only=False)
        
    def setup(self, stage = None):
        pass
        
    def train_dataloader(self):
        return DataLoader(
            dataset=self.trainset,
            batch_size = self.batch_size,
            shuffle= True,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset=self.testset,
            batch_size = self.batch_size,
            shuffle= False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )

if __name__ == '__main__':

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger
    from torchlop import profile
    from sklearn.metrics import confusion_matrix
    import seaborn as sn
    import pandas as pd
    import matplotlib.pyplot as plt

    num_classes = 12
    batch_size = 64

    # Choose model name here to reproduce accuracy results on raw audio dataset
    model_name = "M3"  # "M3" or "M4"

    pytorch_model = \
        M3(
            n_output    = num_classes,
            n_channel   = 64
        ) \
        if model_name == "M3" else \
        M4(
            n_output    = num_classes,
            n_channel   = 64
        )

    macs, params, layer_infos = profile(pytorch_model, inputs=(torch.empty(1, 64, 496),)) 
    # for layer_name, (layer_mac, layer_params) in layer_infos.items(): 
    #     print(f"{layer_name}: MACs = {layer_mac}, Parameters = {layer_params}")
        
    LitModel = LitClassifier(
                model           = pytorch_model,
                num_classes     = num_classes,
                max_lr          = 1e-2,
    )

    callbacks = [
        ModelCheckpoint(
                        save_top_k=1, 
                        mode='max', 
                        monitor="valid_acc"
                    )
    ]  # save top 1 model 
    # logger = CSVLogger(save_dir="logs/dnpu_logs/", name="analog_trained" if analog_train else "full_precision")
    # logger_tb = TensorBoardLogger("logs/dnpu_logs/tb_logs", name="analog_trained" if analog_train else "full_precision")

    trainer = L.Trainer(
        max_epochs  = 100,
        callbacks   = callbacks,
        accelerator ='auto',
        devices     = 'auto',
        # logger      = [logger, logger_tb],
        # log_every_n_steps   = 20,
        # precision = 16
    )

    trainer.fit(model=LitModel, datamodule=LitDataModule(num_classes=num_classes, batch_size=batch_size))

    print("Checking best model accuracy...")
    best_model_path = callbacks[0].best_model_path
    model = LitClassifier.load_from_checkpoint(
        best_model_path, 
        model = pytorch_model,
        num_classes     = num_classes,
    )
    model.eval()
    validation_dataloader = trainer.val_dataloaders
    accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    all_preds = []
    all_labels = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    with torch.no_grad():
        for batch in validation_dataloader:
            features, true_labels = batch
            logits = (model(features.to(device)).squeeze()).to(device)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds)
            all_labels.append(true_labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = accuracy_metric(all_preds.to('cpu'), all_labels.to('cpu'))
    print(f'Best Validation Accuracy: {accuracy.item()}')

    # plot confusion matrix
    classes = ['Unknown', 'Down', 'Go', 'Left', 'No', 'Off', 'On', 'Right',
        'Stop', 'Up', 'Yes', 'House']
    cf_matrix = confusion_matrix(all_labels.detach().cpu(), all_preds.detach().cpu())
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                     columns = [i for i in classes])
    plt.figure();sn.heatmap(df_cm, annot=True);plt.show(block=True)
