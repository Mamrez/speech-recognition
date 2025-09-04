import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader

torch.set_float32_matmul_precision('medium')

import pytorch_lightning as L
import torchmetrics

import time

from pytorch_lightning.loggers import TensorBoardLogger

from aihwkit_lightning.optim import AnalogOptimizer

class M4(nn.Module):
    def __init__(self, n_output = 12, n_channel = 64):
        super().__init__()
        self.layer_norm = nn.BatchNorm1d(n_channel)
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
    def __init__(self, *args, **kwargs) -> None:
        super().__init__()

        self.model = kwargs.get('model')
        self.num_classes = kwargs.get('num_classes')
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
        self.train_acc(predicted_labels, true_labels)
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
                torch.optim.AdamW,
                lambda: list(self.model.analog_layers()),
                self.parameters()
            )
        else:
            optimizer = torch.optim.AdamW(self.parameters())

        return optimizer
    
class LitDataModule(L.LightningDataModule):
    def __init__(self, num_classes, batch_size = 32, num_workers = 2) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.batch_size = batch_size
        self.num_workers = num_workers
        

    def prepare_data(self):
        pass
        
    def setup(self, stage = None):
        self.trainset = torch.load("GSC/datasets/dnpu_measurements/trainset_kernel=8_12classes.pt", weights_only=False)
        self.testset = torch.load("GSC/datasets/dnpu_measurements/testset_kernel=8_12classes.pt", weights_only=False)
    
    def train_dataloader(self):
        return DataLoader(
            dataset         = self.trainset,
            batch_size      = self.batch_size,
            shuffle         = True,
            drop_last       = False,
            num_workers     = self.num_workers,
            persistent_workers  = True
        )
    
    def val_dataloader(self):
        return DataLoader(
            dataset         = self.testset,
            batch_size      = self.batch_size,
            shuffle         = False,
            drop_last       = False,
            num_workers     = self.num_workers,
            persistent_workers = True
        )

if __name__ == '__main__':

    from pytorch_lightning.callbacks import ModelCheckpoint
    from pytorch_lightning.loggers import CSVLogger

    from aihwkit_lightning.nn.conversion import convert_to_analog

    from aihwkit_lightning.simulator.configs import (
        TorchInferenceRPUConfig,
        WeightClipType,
        WeightNoiseInjectionType,
    )

    # Boolean flag to enable/disable model checkpointing 
    MODEL_CHECKPOINTING = False

    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.inp_res = 2**8 - 2
    rpu_config.forward.out_noise = 0.1
    rpu_config.forward.out_noise_per_channel = True
    rpu_config.forward.out_bound = -1
    rpu_config.forward.out_res = -1

    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL
    rpu_config.clip.sigma = 1.5

    rpu_config.modifier.noise_type = WeightNoiseInjectionType.ADD_NORMAL_PER_CHANNEL

    rpu_config.modifier.std_dev = 0.1

    rpu_config.mapping.max_input_size = -1

    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.init_from_data = 100
    rpu_config.pre_post.input_range.init_std_alpha = 3.0
    rpu_config.pre_post.input_range.decay = 0.15

    num_classes = 12
    batch_size = 256
    analog_train = True

    pytorch_model = M4(
        n_output=num_classes,
        n_channel=64
    )
        
    LitModel = LitClassifier(
                model           = convert_to_analog(pytorch_model, rpu_config, verbose=True) if analog_train else pytorch_model,
                num_classes     = num_classes,
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
        callbacks   = callbacks if MODEL_CHECKPOINTING else None,
        accelerator ='gpu',
        devices     = [0],
        logger      = [logger, logger_tb] if MODEL_CHECKPOINTING else None,
        log_every_n_steps   = 20,
        precision = '16-mixed'
    )

    dm = LitDataModule(num_classes=num_classes, batch_size=batch_size)
    start_time = time.time()
    trainer.fit(model=LitModel, datamodule=dm)

    runtime = (time.time() - start_time)/60
    print(f"Training took {runtime:.2f} min in total.")

