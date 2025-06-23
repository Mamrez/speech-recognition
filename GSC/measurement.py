# import sklearn.preprocessing
# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# import torch.optim as optim
# import torchaudio
# import sys
import os
import librosa
# import sklearn
# import torchaudio

# from torch.utils.data import DataLoader, Dataset, random_split

# import matplotlib.pyplot as plt
# import IPython.display as ipd
import numpy as np

from tqdm import tqdm
from itertools import chain
# from torchvision import transforms

from brainspy.utils.manager import get_driver

def set_random_control_voltages( 
                meas_input,
                dnpu_control_indeces,
                slope_length,
                projection_idx,
                rand_matrix #  (len(dnpu_control_indeces), 128) -> ((6, 128))
                ):
    for i in range(len(dnpu_control_indeces)):
        ramp_up = np.linspace(0, rand_matrix[i, projection_idx], slope_length)
        plateau = np.linspace(rand_matrix[i, projection_idx], rand_matrix[i, projection_idx], np.shape(meas_input)[1] - 2 * slope_length)
        ramp_down = np.linspace(rand_matrix[i, projection_idx], 0, slope_length)
        meas_input[dnpu_control_indeces[i], :] = np.concatenate((ramp_up, plateau, ramp_down))

    return meas_input

def measurement(
    configs,
    n_output_channels, # number of dnpu configs
    slope_length,
    rest_length,
    dnpu_input_index,
    dnpu_control_indeces
):

    # dataset_np = np.load("dataset/SUBSET/numpy_audios/dataset_np.npy", allow_pickle=True)
    # labels_np = np.load("dataset/SUBSET/numpy_audios/labels_np.npy", allow_pickle=True)

    dataset, label = [], []
    data_dir = "C:/Users/Mohamadreza/OneDrive - University of Twente/Documenten/github/brainspy-tasks/bspytasks/temporal_kernels/GoogleSpeechCommands/dataset/SUBSET/raw_audios/"
    empty = "C:/Users/Mohamadreza/OneDrive - University of Twente/Documenten/github/brainspy-tasks/bspytasks/temporal_kernels/empty"
    for subdir, _, files in chain.from_iterable(
        os.walk(path) for path in (data_dir, empty)
    ):
        for file in files:
            tmp, _ = librosa.load(os.path.join(subdir, file), sr=8000, mono=True, dtype=np.float32)
            dataset.append(tmp)
            label.append(file[0])


    dataset_np = dataset_np[21 * 200 : -1 , :]
    # measure every 7 classes together
    dnpu_output = np.zeros((dataset_np.shape[0], 64, dataset_np.shape[1]))

    driver = get_driver(configs["driver"])

    # Dividing random voltage of neighbouring electrodes by a factor of 2
    rand_matrix = np.random.uniform(
        -0.6, 
        0.6, 
        size = (len(dnpu_control_indeces), n_output_channels)
    )
    for i in range(0, len(rand_matrix)):
        if i == 0 or i == 5:
            rand_matrix[i][:] = rand_matrix[i][:] / 4

    counter_for_driver_reset = 0
    for d in tqdm(range(len(dataset_np)), desc="Measuring training data..."):
        for p_idx in tqdm(range(n_output_channels), desc = "Measuring projection"):
            counter_for_driver_reset += 1
            # dnpu measurement input
            meas_inputs = np.zeros(
                (
                    len(dnpu_control_indeces) + 1,
                    len(dataset_np[d]) + 2 * slope_length + rest_length
                )
            )
            
            meas_inputs[dnpu_input_index,slope_length + rest_length : -slope_length] = dataset_np[d] 
            meas_inputs = set_random_control_voltages(
                meas_input= meas_inputs,
                dnpu_control_indeces= dnpu_control_indeces,
                slope_length= slope_length,
                projection_idx= p_idx,
                rand_matrix= rand_matrix
            )

            output = driver.forward_numpy(meas_inputs.T)
            output = output[slope_length + rest_length : -slope_length, 0]
            output = output - np.mean(output)

            dnpu_output[d][p_idx] = output
            
        np.save("C:/Users/Mohamadreza/Documents/github/brainspy-tasks/bspytasks/temporal_kernels/GoogleSpeechCommands/dataset/dnpu_measurements/dnpu_output_classes_21_22_23_24_25_26_27_28_29.npy", dnpu_output)
        # if counter_for_driver_reset %999 == 0:
        #     driver.close_tasks()
        #     driver = get_driver(configs["driver"])

    driver.close_tasks()
   

if __name__ == '__main__':

    # 
    np.random.seed(0)
    print(np.random.randn(4))

    # # doing measurements
    from brainspy.utils.io import load_configs
    configs = load_configs(
        "C:/Users/Mohamadreza/OneDrive - University of Twente/Documenten/github/brainspy-tasks/configs/defaults/processors/hw_gsc.yaml"
    )
    measurement(
        configs,
        64,
        100,
        400,
        dnpu_input_index= 3,
        dnpu_control_indeces = [0, 1, 2, 4, 5, 6]
    )
