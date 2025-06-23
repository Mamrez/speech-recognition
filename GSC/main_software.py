# from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
import sklearn.preprocessing
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import sklearn
import scipy
from sklearn import model_selection


import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')
# os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"

from torch.utils.data import DataLoader

import numpy as np


import pytorch_lightning as L
import torchmetrics

import time


from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import CSVLogger
     
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class M5(nn.Module):
    def __init__(self, n_input=1, n_output=12, stride=16, n_channel=32):
        super().__init__()
        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv1(x.resize(x.size(0), 1, x.size(1)))
        x = F.relu(self.bn1(x))
        x = self.pool1(x)
        x = self.conv2(x)
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class M4(nn.Module):
    def __init__(self, n_output, n_channel):
        super().__init__()
        self.conv2 = nn.Conv1d(1, n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(2 * n_channel)
        self.pool4 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(2 * n_channel, n_output)

    def forward(self, x):
        x = self.conv2(x.resize(x.size(0), 1, x.size(1)))
        x = F.relu(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.relu(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.relu(self.bn4(x))
        x = self.pool4(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class LitClassifier(L.LightningModule):
    def __init__(self, *args: torch.Any, **kwargs: torch.Any) -> None:
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

        # return optimizer
    
class LitDataModule(L.LightningDataModule):
    def __init__(self, num_classes, batch_size = 32, num_workers = 2, down_sample = False) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.down_sample = down_sample

    def prepare_data(self):
        self.labels_np = np.load("dataset/SUBSET/numpy_audios/labels_np.npy", 
                        allow_pickle=True)
        self.dataset = np.load("dataset/SUBSET/numpy_audios/dataset_np.npy", 
                        allow_pickle=True)
        
        if self.down_sample:
            self.dataset = self.dataset[:, 0:7936:16]
        
        # The following implementation will produce stratified dataset; in 
        # AIMC measurement, we do not use this implementation
        # if self.num_classes == 12:
        #     # rearanging for KWS task
        #     for i in range(len(self.labels_np)):
        #         if self.labels_np[i] not in [4, 8, 11, 14, 15, 16, 18, 22, 26, 28, 10]:
        #             self.labels_np[i] = 0

        #     self.labels_np[np.where(self.labels_np == 4)] = 1
        #     self.labels_np[np.where(self.labels_np == 8)] = 2
        #     self.labels_np[np.where(self.labels_np == 11)] = 3
        #     self.labels_np[np.where(self.labels_np == 14)] = 4
        #     self.labels_np[np.where(self.labels_np == 15)] = 5
        #     self.labels_np[np.where(self.labels_np == 16)] = 6
        #     self.labels_np[np.where(self.labels_np == 18)] = 7
        #     self.labels_np[np.where(self.labels_np == 22)] = 8
        #     self.labels_np[np.where(self.labels_np == 26)] = 9
        #     self.labels_np[np.where(self.labels_np == 28)] = 10
        #     self.labels_np[np.where(self.labels_np == 10)] = 11

        # print("")

        
    def setup(self, stage = None):
        train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(
            self.dataset,
            self.labels_np, 
            test_size = 0.1, 
            train_size = 0.9,
            stratify=self.labels_np,
            random_state = 0
        )

        # preparing for KWS task
        if self.num_classes == 12:
            for i in range(len(train_label)):
                if train_label[i] not in [4, 8, 11, 14, 15, 16, 18, 22, 26, 28, 10]:
                    train_label[i] = 0
            for i in range(len(test_label)):
                if test_label[i] not in [4, 8, 11, 14, 15, 16, 18, 22, 26, 28, 10]:
                    test_label[i] = 0
            train_label[np.where(train_label == 4)] = 1
            train_label[np.where(train_label == 8)] = 2
            train_label[np.where(train_label == 11)] = 3
            train_label[np.where(train_label == 14)] = 4
            train_label[np.where(train_label == 15)] = 5
            train_label[np.where(train_label == 16)] = 6
            train_label[np.where(train_label == 18)] = 7
            train_label[np.where(train_label == 22)] = 8
            train_label[np.where(train_label == 26)] = 9
            train_label[np.where(train_label == 28)] = 10
            train_label[np.where(train_label == 10)] = 11
            test_label[np.where(test_label == 4)] = 1
            test_label[np.where(test_label == 8)] = 2
            test_label[np.where(test_label == 11)] = 3
            test_label[np.where(test_label == 14)] = 4
            test_label[np.where(test_label == 15)] = 5
            test_label[np.where(test_label == 16)] = 6
            test_label[np.where(test_label == 18)] = 7
            test_label[np.where(test_label == 22)] = 8
            test_label[np.where(test_label == 26)] = 9
            test_label[np.where(test_label == 28)] = 10
            test_label[np.where(test_label == 10)] = 11
            
        
        # torch_data_test = torch.empty(size=(len(test_data), 64, 496 if self.down_sample else 8000))
        torch_data_test = torch.empty_like(torch.Tensor(test_data))
        torch_targets_test = torch.empty(size=(len(test_data),))
        for i in range(0, len(test_data)):
            torch_data_test[i] = torch.Tensor(test_data[i])
            torch_targets_test[i] = test_label[i]
        self.testset = torch.utils.data.TensorDataset(torch_data_test, torch_targets_test)
        
        # train set
        # torch_data_train = torch.empty(size=(len(train_data), 64, 496 if self.down_sample else 8000))
        torch_data_train = torch.empty_like(torch.Tensor(train_data))
        torch_targets_train = torch.empty(size=(len(train_data),))
        for i in range(0, len(train_data)):
            torch_data_train[i] = torch.Tensor(train_data[i])
            torch_targets_train[i] = train_label[i]
        self.trainset = torch.utils.data.TensorDataset(torch_data_train, torch_targets_train)
    
    def train_dataloader(self) -> torch.Any:
        return DataLoader(
            dataset=self.trainset,
            batch_size = self.batch_size,
            shuffle= True,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )
    
    def val_dataloader(self) -> torch.Any:
        return DataLoader(
            dataset=self.testset,
            batch_size = self.batch_size,
            shuffle= False,
            drop_last=False,
            num_workers=self.num_workers,
            persistent_workers=True
        )


if __name__ == '__main__':

    from torchlop import profile

    num_classes = 12
    batch_size = 32

    pytorch_model = M4(
        n_output=num_classes,
        n_channel=64
    )

    pytorch_model = M5(
        n_input     = 1,
        n_output    = num_classes,
        n_channel   = 64
    )


    macs, params, layer_infos = profile(pytorch_model, inputs=(torch.empty(1, 8000),)) 
    for layer_name, (layer_mac, layer_params) in layer_infos.items(): 
        print(f"{layer_name}: MACs = {layer_mac}, Parameters = {layer_params}")


    datamodule = LitDataModule(
        num_classes = num_classes,
        batch_size  = batch_size,
        down_sample = False
    )

    LitModel = LitClassifier(
        model = pytorch_model,
        num_classes = num_classes,
        max_lr = 1e-2,
    )

    callbacks = [
        ModelCheckpoint(
                        save_top_k=1, 
                        mode='max', 
                        monitor="valid_acc"
                    )
    ]  # save top 1 model 
    # logger = CSVLogger(save_dir="logs/", name= "full_precision")
    # logger_tb = TensorBoardLogger("logs/tb_logs", name= "full_precision")

    trainer = L.Trainer(
        max_epochs  = 500,
        callbacks   = callbacks,
        accelerator ='auto',
        devices     = 'auto',
        # logger      = [logger, logger_tb],
        # log_every_n_steps   = 100
    )

    start_time = time.time()
    trainer.fit(model=LitModel, datamodule=datamodule)

    runtime = (time.time() - start_time)/60
    print(f"Training took {runtime:.2f} min in total.")

    print("Checking best model accuracy...")
    best_model_path = callbacks[0].best_model_path
    model = LitClassifier.load_from_checkpoint(best_model_path, model = pytorch_model,
                                        num_classes     = num_classes,
                                        max_lr          = 1e-2,
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

