import torch
import torch.nn as nn
import torch.nn.functional as F
import tqdm
from torchlop import profile

from torch.utils.data import DataLoader

import numpy as np

from tqdm import tqdm

class LinearLayer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.bn = nn.LazyBatchNorm1d()
        self.fc = nn.LazyLinear(10)
    
    def forward(self, x):
        x = self.bn(x[:, 0:10, 0:971])
        x = self.fc(x.flatten(1))
        return F.log_softmax(x, dim=1)

class M3Compact(nn.Module):
    def __init__(self, input_ch, n_channels, relu=False) -> None:
        super().__init__()
        self.relu = relu
        # self.bn1 = nn.BatchNorm1d(input_ch)
        self.conv1 = nn.Conv1d(input_ch, 96, kernel_size=8, stride=1)
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.pool1 = nn.MaxPool1d(8)
        self.conv2 = nn.Conv1d(n_channels, 53, kernel_size=3)
        self.bn3 = nn.BatchNorm1d(n_channels)
        self.pool2 = nn.MaxPool1d(4)
        self.conv3 = nn.Conv1d(n_channels, 53, kernel_size=3)
        self.bn4 = nn.BatchNorm1d(53)
        self.pool3 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(53, 10)
 
    def forward(self, x):
        mask = x.ne(0.)
        # x = self.bn1(x)
        # x = F.layer_norm(x, x.shape)
        x = self.conv1(mask * x)
        mask = x.ne(0.)
        x = self.bn2(x)
        if self.relu:
            x = F.relu(x)
        else:
            x = F.tanh(x)
        x = self.pool1(mask * x)
        x = self.conv2(x)
        mask = x.ne(0.)
        x = self.bn3(x)
        if self.relu:
            x = F.relu(x)
        else:
            x = F.tanh(x)
        x = self.pool2(mask * x)
        x = self.conv3(x)
        mask = x.ne(0.)
        x = self.bn4(x)
        if self.relu:
            x = F.relu(x)
        else:
            x = F.tanh(x)
        # x = self.pool3(x)
        x = F.avg_pool1d(mask * x, 36) # x.shape[-1] must be traceable
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class M1Compact(nn.Module):
    def __init__(self, input_ch = 64, n_channels=32) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_ch)
        self.conv1 = nn.Conv1d(input_ch, out_channels = n_channels, kernel_size = 3)
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.pool1 = nn.MaxPool1d(8)
        self.fc1   = nn.Linear(32, 10)
    
    def forward(self, x):
        # the data length is 1250 while the x[971:1250] are zeros. We remove to speed up the process time. This step has no effect on the accuracy.
        x = self.bn1(x[:, 0:16, 0:972])
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.relu(x)
        x = self.pool1(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class M2Compact(nn.Module):
    def __init__(self, input_ch = 64, n_channels=32) -> None:
        super().__init__()
        self.bn1 = nn.BatchNorm1d(input_ch)
        self.conv1 = nn.Conv1d(input_ch, out_channels = n_channels, kernel_size = 3)
        self.bn2 = nn.BatchNorm1d(n_channels)
        self.pool1 = nn.MaxPool1d(8)
        self.conv2 = nn.Conv1d(in_channels = n_channels, out_channels = n_channels, kernel_size = 3)
        self.bn3 = nn.BatchNorm1d(n_channels)
        self.pool2 = nn.MaxPool1d(8)
        self.fc1   = nn.Linear(32, 10)
    
    def forward(self, x):
        # the data length is 1250 while the x[971:1250] are zeros. We remove to speed up the process time. This step has no effect on the accuracy.
        x = self.bn1(x[:, :, 0:972:2])
        x = self.conv1(x)
        x = self.bn2(x)
        x = F.tanh(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = F.tanh(x)
        x = self.pool2(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

class ToTensor(object):
    def __call__(self, sample) -> object:
        audio_data, audio_label = sample['audio_data'], sample['audio_label']
        return {
            'audio_data'        : torch.tensor(audio_data, dtype=torch.float),
            'audio_label'       : torch.tensor(np.asarray(audio_label, dtype=np.float32), dtype=torch.float)
        }
   
def train(
    model, num_epochs, weight_decay, train_loader, test_loader, device, batch_size
):
    model.to(device)
    loss_fn = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(), 
        weight_decay    = weight_decay
    )

    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr          = 0.01,
        steps_per_epoch = int(len(train_loader)),
        epochs          = num_epochs,
        anneal_strategy = 'cos',
        cycle_momentum  = True
    )

    model.train()
    accuracies = [0]
    
    for epoch in range(num_epochs):
        with tqdm(train_loader, unit="batch") as tepoch:
            current_loss = 0.
            for i, data in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")

                inputs = data[0].to(device)
                targets = data[1].type(torch.LongTensor).to(device)
                
                outputs = torch.squeeze(model(inputs))
                loss = loss_fn(outputs, targets)

                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                scheduler.step()
                
                current_loss += loss.item()

                if i % batch_size  == batch_size - 1:
                    current_loss = 0.

                tepoch.set_postfix(loss=current_loss, accuracy = accuracies[-1])

            if epoch >= 40:
                model.eval()
                with torch.no_grad():
                    correct, total = 0, 0
                    for i, data in enumerate(test_loader):
                        inputs = data[0].to(device)
                        targets = data[1].type(torch.LongTensor).to(device)
                        outputs = torch.squeeze(model(inputs))
                        _, predicted = torch.max(outputs, 1)
                        total += data[1].size(0)
                        correct += (predicted == targets).sum().item()

                accuracies.append(100*correct/total)
                model.train()
            

    print("-----------------------------------------")
    print("Best test accuracy: ", np.max(accuracies))


if __name__ == '__main__':

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    batch_size = 16

    sample_shape = (64, 1250)
    
    # - Create dataset
    np_data_test = np.load("speech-recognition/data/testset_numpy.npy", allow_pickle = True)
    np_data_train = np.load("speech-recognition/data/trainset_numpy.npy", allow_pickle = True)

    # test set
    torch_data_test = torch.empty(size=(len(np_data_test),sample_shape[0], sample_shape[1]))
    torch_targets_test = torch.empty(size=(len(np_data_test),))
    for sample_idx, sample in enumerate(np_data_test):
        torch_data_test[sample_idx] = sample["audio_data"]
        torch_targets_test[sample_idx] = sample["audio_label"].long()
    testset = torch.utils.data.TensorDataset(torch_data_test, torch_targets_test)
    
    # train set
    torch_data_train = torch.empty(size=(len(np_data_train),sample_shape[0], sample_shape[1]))
    torch_targets_train = torch.empty(size=(len(np_data_train),))
    for sample_idx, sample in enumerate(np_data_train):
        torch_data_train[sample_idx] = sample["audio_data"]
        torch_targets_train[sample_idx] = sample["audio_label"].long()
    trainset = torch.utils.data.TensorDataset(torch_data_train, torch_targets_train)

    train_loader = DataLoader(
        trainset,
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = False
    )

    test_loader = DataLoader(
        testset,
        batch_size  = 2,
        shuffle     = False,
        drop_last   = False
    )

    model = M2Compact(
        input_ch = 64,
    )
    macs, params, layer_infos = profile(model, inputs=(torch.empty(1, 64, 972),)) 

    # model = LinearLayer()

    # model = M3Compact(input_ch=64, n_channels=64)
    
    print("Number of learnable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    train(
        model           = model,
        num_epochs      = 500,
        weight_decay    = 1e-4,
        train_loader    = train_loader,
        test_loader     = test_loader,
        device          = device,
        batch_size      = batch_size,
    )

