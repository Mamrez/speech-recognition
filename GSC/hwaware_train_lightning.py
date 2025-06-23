# from lightning.pytorch.utilities.types import TRAIN_DATALOADERS
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

# import librosa
# import matplotlib.pyplot as plt

# import matplotlib.pyplot as plt
# import IPython.display as ipd
import numpy as np

from tqdm import tqdm
from itertools import chain
from torchvision import transforms

import pytorch_lightning as L
import torchmetrics

import time

from pytorch_lightning.loggers import TensorBoardLogger


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def butter_lowpass(cutoff, order, fs):
    return scipy.signal.butter( N = order, 
                                Wn = cutoff, 
                                btype = 'low', 
                                analog = False,
                                fs= fs,
                                output = 'sos'
    )

def butter_lowpass_filter(data, cutoff, order, fs):
    sos = butter_lowpass(cutoff, order = order, fs=fs)
    return scipy.signal.sosfilt(sos, data)

class M4(nn.Module):
    def __init__(self, n_output = 12, n_channel = 64):
        super().__init__()
        self.layer_norm = nn.BatchNorm1d(n_channel)
        # self.layer_norm = nn.LayerNorm((64, 496*4))
        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=8)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channel, 4 * n_channel, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(4 * n_channel)
        self.pool3 = nn.MaxPool1d(4)
        self.conv4 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(4 * n_channel)
        self.pool4 = nn.MaxPool1d(4)

        self.conv_ext = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3)
        self.bn_ext = nn.BatchNorm1d(4 * n_channel)
        self.pool_ext = nn.MaxPool1d(4)

        self.fc1 = nn.Linear(4 * n_channel, n_channel)
        self.fc2 = nn.Linear(n_channel, n_output)

    def forward(self, x):
        x = self.layer_norm(x)

        x = self.conv2(x)
        x = F.tanh(self.bn2(x))
        x = self.pool2(x)
        x = self.conv3(x)
        x = F.tanh(self.bn3(x))
        x = self.pool3(x)
        x = self.conv4(x)
        x = F.tanh(self.bn4(x))
        x = self.pool4(x)

        x = self.conv_ext(x)
        x = F.tanh(self.bn_ext(x))
        x = self.pool_ext(x)

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = F.tanh(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=2)

class LitClassifier(L.LightningModule):
    def __init__(self, *args: torch.Any, **kwargs: torch.Any) -> None:
        super().__init__()

        self.model = kwargs.get('model')
        self.num_classes = kwargs.get('num_classes')
        self.max_lr = kwargs.get('max_lr')
        self.analog_train = kwargs.get('analog_train')

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
        if self.analog_train:
            optimizer = AnalogOptimizer(
                torch.optim.AdamW, self.model.analog_layers(), self.parameters(), lr = 1e-4, weight_decay = 1e-4
            )
        else:
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

        # return [optimizer], [lr_scheduler_config]

        return optimizer
    
class LitDataModule(L.LightningDataModule):
    def __init__(self, num_classes, batch_size = 32, num_workers = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers

    def prepare_data(self):
        # self.labels_np = np.load("dataset/SUBSET/numpy_audios/labels_np.npy", 
        #                 allow_pickle=True)[0 * 200 : self.num_classes * 200]
        # folder_path = "dataset/dnpu_measurements"
        # self.dataset = np.zeros(((self.num_classes * 200, 64, 4*496)))
        # i = 0
        # for filename in sorted(os.listdir(folder_path)):
        #     if filename.endswith("6.npy") or filename.endswith("13.npy") or filename.endswith("20.npy"):
        #         if not filename.endswith("29.npy"):
        #             file_path = os.path.join(folder_path, filename)
        #             file = np.load(file_path, allow_pickle=True)
        #             self.dataset[i * 200 : (i + 7) * 200][:][:] = file[:,:,0:7936:4]
        #             i += 7
        # for filename in sorted(os.listdir(folder_path)):
        #     if  filename.endswith("29.npy"):
        #         file_path = os.path.join(folder_path, filename)
        #         file = np.load(file_path, allow_pickle=True)
        #         self.dataset[i * 200 : (i + 9) * 200 - 1][:][:] = file[:,:,0:7936:4]
        self.trainset = torch.load("dataset/dnpu_measurements/trainset_kernel=8_12classes.pt")
        self.testset = torch.load("dataset/dnpu_measurements/testset_kernel=8_12classes.pt")
        print()
        
    def setup(self, stage = None):
        # train_data, test_data, train_label, test_label = sklearn.model_selection.train_test_split(
        #     self.dataset,
        #     self.labels_np, 
        #     test_size = 0.1, 
        #     train_size = 0.9,
        #     stratify=self.labels_np,
        #     random_state = 0
        # )

        # torch_data_test = torch.empty(size=(len(test_data), 64, 4*496))
        # torch_targets_test = torch.empty(size=(len(test_data),))
        # for i in range(0, len(test_data)):
        #     torch_data_test[i] = torch.Tensor(test_data[i])
        #     torch_targets_test[i] = test_label[i]
        # self.testset = torch.utils.data.TensorDataset(torch_data_test, torch_targets_test)
        
        # # train set
        # torch_data_train = torch.empty(size=(len(train_data), 64, 4*496))
        # torch_targets_train = torch.empty(size=(len(train_data),))
        # for i in range(0, len(train_data)):
        #     torch_data_train[i] = torch.Tensor(train_data[i])
        #     torch_targets_train[i] = train_label[i]
        # self.trainset = torch.utils.data.TensorDataset(torch_data_train, torch_targets_train)

        pass
    
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

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    from aihwkit_lightning.nn.conversion import convert_to_analog
    from aihwkit_lightning.simulator.configs import TorchInferenceRPUConfig

    from aihwkit_lightning.simulator.configs import WeightClipType

    from aihwkit_lightning.simulator.configs import WeightModifierType

    from aihwkit_lightning.optim import AnalogOptimizer

    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.inp_res = 2**8 - 2
    rpu_config.forward.out_noise = 0.1
    rpu_config.forward.out_noise_per_channel = True
    rpu_config.forward.out_bound = -1
    rpu_config.forward.out_res = -1

    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL
    rpu_config.clip.sigma = 1.5

    rpu_config.modifier.type = WeightModifierType.ADD_NORMAL_PER_CHANNEL
    rpu_config.modifier.std_dev = 1.0

    rpu_config.mapping.max_input_size = -1

    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.init_from_data = 100
    rpu_config.pre_post.input_range.init_std_alpha = 3.0
    rpu_config.pre_post.input_range.decay = 0.15

    num_classes = 12
    batch_size = 32
    analog_train = True
    load_pre_trained_model = False

    # load pre-trained model
    if load_pre_trained_model:
        if analog_train:
            checkpoint = torch.load("logs/dnpu_logs/full_precision/version_0/checkpoints/epoch=353-step=59826.ckpt")
            new_state_dict = {}
            for key, value in checkpoint['state_dict'].items():
                new_key = key.replace('model.', '')
                new_state_dict[new_key] = value
            pytorch_model = M4(
                n_output=num_classes,
                n_channel=64
            )
            pytorch_model.load_state_dict(new_state_dict)
        else:
            pytorch_model = M4(
                n_output=num_classes,
                n_channel=64
            )
    else:
        pytorch_model = M4(
            n_output=num_classes,
            n_channel=64
        )
        
    LitModel = LitClassifier(
                model           = convert_to_analog(pytorch_model, rpu_config, verbose=True) if analog_train else pytorch_model,
                num_classes     = num_classes,
                max_lr          = 1e-2,
                analog_train    = analog_train
    )

    callbacks = [
        ModelCheckpoint(
                        save_top_k=1, 
                        mode='max', 
                        monitor="valid_acc"
                    )
    ]  # save top 1 model 
    logger = CSVLogger(save_dir="logs/dnpu_logs/", name="analog_trained" if analog_train else "full_precision")
    logger_tb = TensorBoardLogger("logs/dnpu_logs/tb_logs", name="analog_trained" if analog_train else "full_precision")

    trainer = L.Trainer(
        max_epochs  = 500,
        callbacks   = callbacks,
        accelerator ='auto',
        devices     = 'auto',
        logger      = [logger, logger_tb],
        log_every_n_steps   = 20,
        # precision = 16
    )

    start_time = time.time()
    trainer.fit(model=LitModel, datamodule=LitDataModule(num_classes=num_classes, batch_size=batch_size))

    runtime = (time.time() - start_time)/60
    print(f"Training took {runtime:.2f} min in total.")

    print("Checking best model accuracy...")
    best_model_path = callbacks[0].best_model_path
    model = LitClassifier.load_from_checkpoint(best_model_path, model = convert_to_analog(pytorch_model, rpu_config, verbose=True) if analog_train else pytorch_model,
                                        num_classes     = num_classes,
                                        max_lr          = 1e-2,
                                        analog_train    = analog_train
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

