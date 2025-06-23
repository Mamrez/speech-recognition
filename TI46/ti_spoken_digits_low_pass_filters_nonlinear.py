import scipy.signal
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import tqdm
import scipy
import librosa
import os
import sklearn
import random
# import cupy, cupyx
# from cupyx.scipy import signal

from torchvision import transforms
from tqdm import tqdm
from itertools import chain
from torch.utils.data import Dataset, DataLoader

from scipy.signal import hilbert, chirp

from brainspy.utils.manager import get_driver
from brainspy.utils.io import load_configs

from pycochleagram import cochleagram as cgram

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

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

def my_lpf(input, RC):
    rc_transfer_function = scipy.signal.lti([1], [1/(2*np.pi*RC), 1])
    t = np.arange(0, 9710/12500, 1/12500)
    t_out, y_out, _ = scipy.signal.lsim(rc_transfer_function, input, t, interp=False)

    return y_out

def butter_bandpass(lowcut, highcut, fs, order=4):
    return scipy.signal.butter(order, [lowcut, highcut], fs=fs, btype='band', output = 'sos')

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    sos = butter_bandpass(lowcut, highcut, fs, order=order)
    y = scipy.signal.sosfilt(sos, data)
    # y = scipy.signal.lfilter(b, a, data)
    return y


def nonlinear_bandpass(input, center_freq, sr = 12500):
    # # return butter_bandpass_filter(input, float(format(center_freq, '.2f')), float(format(center_freq, '.2f')) + 15, fs = sr, order = 4)[::10]
    # bws = [5, 10, 15, 20, 25, 30, 35, 40, 45, 50]
    # filtered_data = [butter_bandpass_filter(input, float(format(center_freq, '.2f')), float(format(center_freq, '.2f')) + bw, fs = sr, order = 4) for bw in bws]

    # output = np.zeros_like(input)
    # max_range = 0.75
    # for i in range(len(input)):
    #     ratio = np.abs(input[i]) / max_range
    #     ratio *= 10
    #     if int(ratio) > 9 : 
    #         ratio = 9
    #     elif int(ratio) < 1:
    #         ratio = 0
    #     output[i] = filtered_data[int(ratio)-10][i]
    
    output = cgram.cochleagram(
        input, sr=sr, n = 64, low_lim = 10, hi_lim = 625, sample_factor=1
    )

    return output[::10]

def low_pass_nonlinear_time_domain(input, RC, sr = 12500):
    dt = 1 / sr
    alpha = dt / (RC + dt)
    output = np.zeros_like(input)
    output[0] = alpha * input[0]
    for i in range(1, len(input)):
        output[i] = output[i - 1] + alpha * (input[i] - output[i - 1]) + alpha/2 * (input[i-1] - output[i - 2])
    
    return output

def mixed_nonlinear_lowpass(signal, cutoff_freq):
    # Compute the frequency response
    freq_bins = np.fft.rfftfreq(len(signal), d=1/12500)
    
    filtered_signal = butter_lowpass_filter(
        data = signal,
        cutoff= cutoff_freq,
        order= 1,
        fs = 12500
    )

    freq_response_filtered_signal = np.fft.rfft(filtered_signal)

    # loop over first 100 bins of the frequencies
    for f in range(1, 100):
        freq_response_filtered_signal[f] = 0.9 * freq_response_filtered_signal[f]
        # loop over 5 harmonics
        for h in range(2, 7):
            freq_response_filtered_signal[np.where(freq_bins == freq_bins[f] * h)] += (1/h) * freq_response_filtered_signal[f]
    # subharmonics did not work well
    for f in range(100, 250):
        for s_h in range(2, 7):
            freq_response_filtered_signal[np.where(freq_bins == freq_bins[f] / s_h)] += (1/s_h) * freq_response_filtered_signal[f]

    time_domain_output = np.real(np.fft.irfft(freq_response_filtered_signal))[::10]
    
    return time_domain_output

def moving_average(a, n=3):
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def load_audio_dataset(
    data_dir            = None,
    min_max_scale       = True,
    low_pass_filter     = True,
    same_size_audios    = True,
    envelope            = True, 
    nonlinear_filters   = True,
    time_domain_nonlinear_filter = True
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
        
        if envelope == None:
            return dataset_numpy, label_numpy
        
        elif envelope == False:
            if nonlinear_filters == True:
            # low pass filters
            # data after low pass filtering are concatenated
                if time_domain_nonlinear_filter == False:
                    dataset_numpy_low_pass = np.zeros((len(dataset), 64 * max_length // 10))
                    rand_matrix_of_cutoffs = np.linspace(20, 625, num=64)
                    # rand_matrix_of_RCs = np.linspace(34e-3, 12e-3, num=64)
                    rand_matrix_of_bandpass = np.linspace(45, 145, num=64)
                    for i in range(len(dataset_numpy)):
                        for j in range(0, 64):
                            # lowpass_data = butter_bandpass_filter(
                            #     dataset_numpy[i],
                            #     float(format(rand_matrix_of_cutoffs[j], '.2f')),
                            #     float(format(rand_matrix_of_cutoffs[j], '.2f')) + 35,
                            #     fs = 12500,
                            #     order = 4
                            # )[::10]
                            # lowpass_data = butter_lowpass_filter(
                            #     dataset_numpy[i],
                            #     float(format(rand_matrix_of_cutoffs[j], '.2f')),
                            #     order = 1,
                            #     fs = 12500
                            # )[::10]
                            # if j == 0:
                            #     lowpass_data = mixed_nonlinear_lowpass(dataset_numpy[i],float(format(rand_matrix_of_cutoffs[j], '.2f')))
                            # else:
                            #     lowpass_data = mixed_nonlinear_lowpass(dataset_numpy[i],float(format(rand_matrix_of_cutoffs[j], '.2f')))
                            #     prev_filt = np.zeros((10 + len(lowpass_data)))
                            #     prev_filt[10:] = 0.3 * dataset_numpy_low_pass[i, max_length//10 * (j-1) : max_length//10 * j]
                            #     lowpass_data += prev_filt[:-10]
                            
                            # # adding some bandpass effects
                            # bandpass_data = butter_bandpass_filter(
                            #     dataset_numpy[i], 
                            #     float(format(rand_matrix_of_bandpass[j], '.2f')),
                            #     float(format(rand_matrix_of_bandpass[j], '.2f')) + 35,
                            #     fs = 12500,
                            #     order = 4
                            # )
                            lowpass_data = nonlinear_bandpass(dataset_numpy[i], rand_matrix_of_cutoffs[j])
                            dataset_numpy_low_pass[i, max_length//10 * j : max_length//10 * (j+1)] = lowpass_data # + 0.2 * np.max(np.abs(lowpass_data)) * bandpass_data[::10]
                    return dataset_numpy_low_pass, label_numpy
                else:
                    ataset_numpy_low_pass = np.zeros((len(dataset), 64 * max_length // 10))
                    rand_matrix_of_cutoffs = np.linspace(20, 625, num=64)
                    for i in range(len(dataset_numpy)):
                        for j in range(0, 64):                             
                            dataset_numpy_low_pass[i, max_length//10 * j : max_length//10 * (j+1)] = low_pass_nonlinear_time_domain(
                                dataset_numpy[i], 1 / (2 * np.pi * rand_matrix_of_cutoffs[j])
                            )
                    return dataset_numpy_low_pass, label_numpy
            else:
                # low pass filters
                # data after low pass filtering are concatenated
                dataset_numpy_low_pass = np.zeros((len(dataset), 64 * max_length // 10))
                RCs = np.linspace(12e-3, 34e-3, num = 64)
                fs = 12500
                for i in range(len(dataset_numpy)):
                    for j in range(0, len(RCs)):
                        X_w = np.fft.rfft(dataset_numpy[i])
                        freqs = np.fft.rfftfreq(len(dataset_numpy[i]), 1/fs)
                        Lf_w = 1 / (1 + freqs / RCs[j])
                        multiplication_res = Lf_w * X_w # * (1 / (1 + np.exp(-np.abs(X_w))))
                        dataset_numpy_low_pass[i, max_length//10 * j : max_length//10 * (j+1)] = np.fft.irfft(multiplication_res)[::10]
                # for i in range(len(dataset_numpy)):
                #     y = np.zeros((dataset_numpy.shape[1]))
                #     for j in range(0, 64):
                #         alpha = (1/fs) / (RCs[j] + 1/fs)
                #         y[0] = alpha * dataset_numpy[i][0]
                #         for k in range(1, len(dataset_numpy[i])):
                #             # alpha *= 1 / (1 + np.exp(dataset_numpy[i][k] ** 2)) not working
                #             y[k] = y[k - 1] + alpha * (dataset_numpy[i][k] - y[k-1]) 
                #         dataset_numpy_low_pass[i, max_length//10 * j : max_length//10 * (j+1)] = y[::10]
                return dataset_numpy_low_pass, label_numpy
        else:
            # calculating envelope
            dataset_numpy_envelope = np.zeros((len(dataset), max_length//10))
            for i in range(len(dataset_numpy)):
                # method 1
                dataset_numpy_envelope[i] = butter_lowpass_filter(
                    np.abs(hilbert(dataset_numpy[i])), cutoff = 30, fs=12500, order=3
                )[::10]

                # method 2
                # dataset_numpy_envelope[i] = np.abs(hilbert(dataset_numpy[i]))[::10]

                # method 3
                # dataset_numpy_envelope[i] = np.abs(hilbert(
                #     butter_lowpass_filter(dataset_numpy[i], cutoff = 60, fs=12500, order=3)
                # ))[::10]

            return dataset_numpy_envelope, label_numpy
    
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

class OnlyLinearLayer(nn.Module):
    def __init__(self) -> None:
        super(OnlyLinearLayer, self).__init__()
        # self.ln = nn.LayerNorm(971)  # 62144
        self.ln = nn.LazyBatchNorm1d()
        # self.linear_layer = nn.Linear(971, 10)
        self.linear_layer = nn.LazyLinear(10)
    def forward(self, x):
        x = self.ln(x)
        x = self.linear_layer(x)
        return F.log_softmax(x, dim=1)

class ConvNet(nn.Module):
    def __init__(self, n_input = 1, n_output=10, n_channel = 32):
        super().__init__()
        self.n_input = n_input
        self.bn0 = nn.BatchNorm1d(n_input)
        self.conv1 = nn.Conv1d(in_channels= self.n_input, out_channels=n_channel, kernel_size=3)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(8)
        self.conv2 = nn.Conv1d(in_channels= n_channel, out_channels=n_channel, kernel_size=3)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)
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
        # lr              = 0.0008, 
        weight_decay    = 1e-3
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
        envelope            = False,
        nonlinear_filters   = True,
        time_domain_nonlinear_filter = False
    )

    # model = OnlyLinearLayer()
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

    print("hi")
    _ = train (
        model.to(device),
        num_epochs      = 500,
        train_loader    = train_loader,
        test_loader     = test_loader,
        save            = False,
        DNPU_train_enabled = False
    )