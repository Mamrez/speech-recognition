import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

import tqdm
import scipy
import scipy.signal
from scipy.signal import gammatone, sosfilt
import librosa
import os
import sklearn

from torchvision import transforms
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset, DataLoader

# from gammatone.gtgram import gtgram

import matplotlib
matplotlib.use('TkAgg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


tanh_params = [[], []]

param_c = 0.0259
param_k = 0.0207

def butter_lowpass(cutoff, order, fs):
    return scipy.signal.butter( N = order, 
                                Wn = cutoff, 
                                btype = 'low', 
                                analog = False,
                                fs= fs,
                                output = 'sos'
    )

def butter_lowpass_filter(data, cutoff, order, fs):
    # b, a = butter_lowpass(cutoff, order = order, fs=fs)
    # y = scipy.signal.filtfilt(b = b, 
    #                         a = a, 
    #                         x = data
    # )
    # return y

    # sos
    sos = butter_lowpass(cutoff, order = order, fs=fs)
    return scipy.signal.sosfilt(sos, data)

    # b, a = butter_lowpass(cutoff, order = order, fs=fs)
    # y = scipy.signal.filtfilt(b = b, 
    #                         a = a, 
    #                         x = data
    # )
    # return y

def load_audio_dataset(
    data_dir            = None,
    min_max_scale       = True,
    low_pass_filter     = True,
    same_size_audios    = True
):
    dataset, label = [], []
    max_length = 0

    for subdir, _, files in chain.from_iterable(
        os.walk(path) for path in data_dir
    ):
        for file in files:
            # Loading audio file;
            # First performing low pass filtering, and then trimming
            tmp, sr = librosa.load(os.path.join(subdir, file), sr=None, mono=True, dtype=np.float32)
            # Amplification, to be chosen in accordance to DNPU performance
            if min_max_scale == True:
                scale = np.max(np.abs(tmp))
                tmp = tmp * (1/scale) * 0.75
            if low_pass_filter == True:
                tmp = butter_lowpass_filter(
                    tmp, 5000, 3, sr
                )
            # Removing silence
            tmp, _ = librosa.effects.trim(
                y               = tmp, 
                top_db          = 12,
                ref             = np.max,
                frame_length    = 128, 
                hop_length      = 4
            )
            
            if len(tmp) > max_length:
                max_length = len(tmp)
                if max_length % 10 != 0:
                    max_length += (10 - (max_length % 10))

            dataset.append(tmp)
            # CAREFUL!!!
            label.append(file[1])

    if same_size_audios == None:
        return dataset, label
    elif same_size_audios == "MAX":
        dataset_numpy = np.zeros((len(dataset), max_length))
        label_numpy = np.zeros((len(dataset)))
        for i in range(len(dataset)):
            dataset_numpy[i][0:len(dataset[i])] = dataset[i]
            label_numpy[i] = label[i]
        
        dataset_filtered = np.zeros((len(dataset_numpy), 64, 971))


        # calculate gammatone filterbanks
        freqs = np.linspace(10, 625, 64)
        # Or choose frequencies to be non-linearly distributed
        # freqs = [ 20, 24.92, 29.93 ,35.05 ,40.26 ,45.58 ,51.00 ,56.53 ,62.17 ,67.93 ,73.79 ,79.77 ,85.87 ,92.09 ,98.43  
        #         , 104.90 ,111.50 ,118.22 ,125.08 ,132.08 ,139.21 ,146.48 ,153.90 ,161.46 ,169.18 ,177.04 ,185.07 ,193.25  
        #         , 201.59  ,210.09 ,218.77  ,227.61  ,236.64  ,245.83  ,255.22  ,264.78 ,274.54  ,284.49 ,294.63 ,304.98   
        #         , 315.53  ,326.28  ,337.26  ,348.44  ,359.85  ,371.49  ,383.35  ,395.45  ,407.79  ,420.37  ,433.20 ,446.28   
        #         , 459.63  ,473.23  ,487.11  ,501.26  ,515.69  ,530.40  ,545.41  ,560.71  ,576.31  ,592.23 ,608.45  ,625]
        for i in range(len(dataset_numpy)):
            for j in range(0, 64):
                b, a = gammatone(freqs[j], 'fir', fs=12500)
                dataset_filtered[i, j] = 0.5 * np.tanh(
                    (scipy.signal.lfilter(b, a, dataset_numpy[i] * 50)) + 1
                )[::10]

                # dataset_filtered[i, j] = scipy.signal.lfilter(b, a, dataset_numpy[i])[::10]
        
        return dataset_filtered, label_numpy
    
class ToTensor(object):
    def __call__(self, data, label) -> object:
        return torch.tensor(data, dtype=torch.float), torch.tensor(np.asarray(label, dtype=np.float32), dtype=torch.float)

class AudioDataset(Dataset):
    def __init__(self, audios, labels, transforms) -> None:
        super(AudioDataset, self).__init__()
        self.transform = transforms
        self.audios = audios
        self.labels = labels
        assert len(self.audios) == len(self.labels), "Error in loading dataset!"
    
    def __len__(self):
        return len(self.audios)

    def __targets__(self):
        return self.labels

    def __getitem__(self, index):
        if isinstance(index, torch.Tensor):
            index = index.tolist()
        
        if self.transform:
            data, label = self.transform(self.audios[index], self.labels[index])
            return data, label
        else:
            return self.audios[index], self.labels[index]

class ConvNet(nn.Module):
    def __init__(self, n_input = 64, n_output=10, n_channel = 32):
        super().__init__()
        self.n_input = n_input
        self.bn0 = nn.BatchNorm1d(n_input)
        self.conv1 = nn.Conv1d(in_channels= self.n_input, out_channels=n_channel, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(8)
        # self.conv2 = nn.Conv1d(in_channels= n_channel, out_channels=n_channel, kernel_size=3)
        # self.bn2 = nn.BatchNorm1d(n_channel)
        # self.pool2 = nn.MaxPool1d(4)
        self.fc1 = nn.Linear(n_channel, n_output)

    def forward(self, x):
        x = x.reshape(x.size(0), self.n_input, 971)
        x = self.bn0(x)
        x = self.conv1(x)
        x = F.tanh(self.bn1(x))
        x = self.pool1(x)
        # x = self.conv2(x)
        # x = F.tanh(self.bn2(x))
        # x = self.pool2(x)
        x = F.avg_pool1d(x, x.shape[-1])
        x = x.permute(0, 2, 1)
        x = self.fc1(x)
        return F.log_softmax(x, dim=2)

def train(
        model,
        num_epochs,
        train_loader,
        test_loader,
        save = True,
        DNPU_train_enabled = True
):
    LOSS = []
    accuracies = [0]
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()), 
        weight_decay    = 1e-4
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, 
        max_lr          = 0.01,
        steps_per_epoch = int(len(train_loader)),
        epochs          = num_epochs,
        anneal_strategy = 'cos',
        cycle_momentum  = True
    )

    for epoch in range(num_epochs):
        if epoch != 0:
            # model.eval()
            model = model.to(device)
            with torch.no_grad():
                correct, total = 0, 0
                for i, (data, label) in enumerate(test_loader):
                    label = label.type(torch.LongTensor).to(device)
                    output = torch.squeeze(model(data.to(device)))
                    _, predicted = torch.max(output, 1)
                    total += label.size(0)
                    correct += (predicted == label).sum().item()
                accuracies.append(100*correct/total)  
        model.train()
        with tqdm(train_loader, unit="batch") as tepoch:
            current_loss = 0
            model.to(device)
            for i, (data, label) in enumerate(tepoch):
                tepoch.set_description(f"Epoch {epoch}")
                label = label.type(torch.LongTensor).to(device)
                output = torch.squeeze(model(data.to(device)))
                loss = loss_fn(output, label)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                # clamping DNPU control voltages
                # DNPUControlVoltageClamp(model, -0.30, 0.30)
                current_loss += loss.item()
                tepoch.set_postfix(
                    loss = current_loss / (i + 1),
                    accuracy = accuracies[-1]
                )
                LOSS.append(current_loss / (i + 1))     
        scheduler.step() 

    return model.state_dict()


if __name__ == "__main__":
    EMPTY = "C:/Users/Mohamadreza/Documents/github/brainspy-tasks/tmp/projected_in_house_arsenic/empty/"
    batch_size = 16
    audios, labels = load_audio_dataset(
        data_dir        = (EMPTY, "C:/Users/Mohamadreza/Documents/ti_spoken_digits/female_speaker"),
        min_max_scale   = True,
        low_pass_filter = True,
        # same_size_audios: can be "NONE" or an "MAX"
        # None -> keep every audio as what it is
        # "MAX" -> extend to maximum audio
        # if "MAX" is chosen, data is returned as numpy arrays, otherwise as list
        same_size_audios    = "MAX",
    )

    model = ConvNet(n_input=64)
    print("Number of learnable params: ", sum(p.numel() for p in model.parameters() if p.requires_grad))

    dataset = AudioDataset(
        audios      = audios,
        labels      = labels,
        transforms  = ToTensor()
    )

    train_idx, test_idx = sklearn.model_selection.train_test_split(
        np.arange(dataset.__len__()),
        test_size       = .1,
        random_state    = 7,
        shuffle         = True,
        stratify        = dataset.__targets__()
    )

    # Subset dataset for train and val
    trainset = torch.utils.data.Subset(dataset, train_idx)
    testset = torch.utils.data.Subset(dataset, test_idx)

    train_loader = DataLoader(
        trainset,
        batch_size  = batch_size,
        shuffle     = True,
        drop_last   = True
    )

    test_loader = DataLoader(
        testset,
        batch_size  = batch_size,
        shuffle     = False,
        drop_last   = True
    )

    _ = train (
        model.to(device),
        num_epochs      = 500,
        train_loader    = train_loader,
        test_loader     = test_loader,
        save            = False,
        DNPU_train_enabled = False
    )
