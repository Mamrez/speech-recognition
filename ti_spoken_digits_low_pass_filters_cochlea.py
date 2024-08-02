import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

import tqdm
import scipy
import scipy.signal
from scipy.signal import gammatone, sosfilt
import librosa
from librosa import display
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
from pycochleagram import erbfilter as erb
from pycochleagram import utils

from gammatone.gtgram import gtgram

import matplotlib
matplotlib.use('TkAgg')

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


tanh_params = [[], []]

param_c = 0.0259
param_k = 0.0207

class NonlinearTanh(nn.Module):
    def __init__(self):
        super().__init__()
        # self.c = nn.Parameter(torch.tensor(0.5))
        # self.k = nn.Parameter(torch.tensor(0.25))

    def forward(self, x): 
        return 0.5 * F.tanh(((1/0.02)*(x-0.03))+1) 
    
    # def forward(self, x):
    #     tanh_params[0].append(self.c)
    #     tanh_params[1].append(self.k)
    #     return 0.5*(torch.tanh((1/self.k)*(x-self.c))+1)
    
    # def _return_params(self):
    #     return self.params

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

def make_harmonic_stack(f0=100, n_harm=40, dur=0.25001, sr=20000, low_lim=50, hi_lim=20000, n=None):
  """Synthesize a tone created with a stack of harmonics.

  Args:
    f0 (int, optional): Fundamental frequency.
    n_harm (int, optional): Number of harmonics to include.
    dur (float, optional): Duration, in milliseconds. Note, the default value
      was chosen to create a signal length that is compatible with the
      predefined downsampling method.
    sr (int, optional): Sampling rate.
    low_lim (int, optional): Lower limit for filterbank.
    hi_lim (int, optional): Upper limit for filerbank.
    n (None, optional): Number of filters in filterbank.

  Returns:
    tuple:
      **signal** (array): Synthesized tone.
      **signal_params** (dict): A dictionary containing all of the parameters
        used to synthesize the tone.
  """
  # i don't know where this came from, but choose a number of filters
  if n is None:
    n = int(np.floor(erb.freq2erb(hi_lim) - erb.freq2erb(low_lim)) - 1)

  # synthesize tone from harmonic stack
  t = np.arange(0, dur + 1 / sr, 1 / sr)
  signal = np.zeros_like(t)
  for i in range(1, n_harm + 1):
    signal += np.sin(2 * np.pi * f0 * i * t)  # zero-phase

  # store all the params in a dictionary
  signal_params = {
      'f0': f0,
      'n_harm': n_harm,
      'dur': dur,
      'sr': sr,
      'low_lim': low_lim,
      'hi_lim': hi_lim,
      'n': n
  }

  return signal, signal_params

def demo_human_cochleagram_helper(signal, sr, n, sample_factor=2, downsample=None, nonlinearity=None):
  """Demo the cochleagram generation.

    signal (array): If a time-domain signal is provided, its
      cochleagram will be generated with some sensible parameters. If this is
      None, a synthesized tone (harmonic stack of the first 40 harmonics) will
      be used.
    sr: (int): If `signal` is not None, this is the sampling rate
      associated with the signal.
    n (int): number of filters to use.
    sample_factor (int): Determines the density (or "overcompleteness") of the
      filterbank. Original MATLAB code supported 1, 2, 4.
    downsample({None, int, callable}, optional): Determines downsampling method to apply.
      If None, no downsampling will be applied. If this is an int, it will be
      interpreted as the upsampling factor in polyphase resampling
      (with `sr` as the downsampling factor). A custom downsampling function can
      be provided as a callable. The callable will be called on the subband
      envelopes.
    nonlinearity({None, 'db', 'power', callable}, optional): Determines
      nonlinearity method to apply. None applies no nonlinearity. 'db' will
      convert output to decibels (truncated at -60). 'power' will apply 3/10
      power compression.

    Returns:
      array:
        **cochleagram**: The cochleagram of the input signal, created with
          largely default parameters.
  """
  human_coch = cgram.human_cochleagram(signal, sr, n=n, sample_factor=sample_factor,
      downsample=downsample, nonlinearity=nonlinearity, strict=False)
  img = np.flipud(human_coch)  # the cochleagram is upside down (i.e., in image coordinates)
  return img


### Waveform Generation from Cochleagram (Inversion) ###
def demo_invert_cochleagram(signal=None, sr=None, n=None, playback=False):
  """Demo that will generate a cochleagram from a signal, then invert this
  cochleagram to produce a waveform signal.

  Args:
    signal (array, optional): Signal containing waveform data.
    sr (int, optional): Sampling rate of the input signal.
    n (int, optional): Number of filters to use in the filterbank.
    playback (bool, optional): Determines if audio signals will be played
      (using pyaudio). If False, only plots will be created. If True, the
      original signal and inverted cochleagram signal will be played.

  Returns:
    None
  """
  # get a signal if one isn't provided
  if signal is None:
    signal, signal_params = make_harmonic_stack()
    sr = signal_params['sr']
    n = signal_params['n']
    low_lim = signal_params['low_lim']
    hi_lim = signal_params['hi_lim']
  else:
    assert sr is not None
    assert n is not None
    low_lim = 50  # this is the default for cochleagram.human_cochleagram
    hi_lim = 20000  # this is the default for cochleagram.human_cochleagram

  # generate a cochleagram from the signal
  sample_factor = 2  # this is the default for cochleagram.human_cochleagram
  coch = demo_human_cochleagram_helper(signal, sr, n, sample_factor=sample_factor)
  print('Generated cochleagram with shape: ', coch.shape)

  # invert the cochleagram to get a signal
  coch = np.flipud(coch)  # the ouput of demo_human_cochleagram_helper is flipped
  inv_coch_sig, inv_coch = cgram.invert_cochleagram(coch, sr, n, low_lim, hi_lim, sample_factor, n_iter=10, strict=False)

  return inv_coch

    
#   print('Generated inverted cochleagram')
#   print('Original signal shape: %s, Inverted cochleagram signal shape: %s' % (signal.shape, inv_coch_sig.shape))

#   plt.subplot(211)
#   plt.title('Cochleagram of original signal')
#   utils.cochshow(coch, interact=False)  # this signal is already flipped
#   plt.ylabel('filter #')
#   plt.xlabel('time')
#   plt.gca().invert_yaxis()

#   plt.subplot(212)
#   plt.title('Cochleagram of inverted signal')
#   utils.cochshow(inv_coch, interact=False)  # this signal needs to be flipped
#   plt.ylabel('filter #')
#   plt.xlabel('time')
#   plt.gca().invert_yaxis()
#   plt.tight_layout()
#   plt.show()


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
 

        # # calculate spectrogram with librosa
        # for i in range(len(dataset_numpy)):
        #     spectrogram = librosa.feature.melspectrogram(
        #         y=dataset_numpy[i],
        #         sr = 12500,
        #     )
        #     re_audio = librosa.feature.inverse.mel_to_audio(spectrogram, sr = 12500)
        
        #     print()

        # for i in range(len(dataset_numpy)):
        #     dataset_filtered[i] = demo_invert_cochleagram(dataset_numpy[i], sr=12500, n = 64)[:, ::10]

    
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

    
    _ = train (
        model.to(device),
        num_epochs      = 500,
        train_loader    = train_loader,
        test_loader     = test_loader,
        save            = False,
        DNPU_train_enabled = False
    )

    print("hi")
