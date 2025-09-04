import torch
import torch.nn as nn
import torch.nn.functional as F
# import matplotlib
# import matplotlib.pyplot as plt
import numpy as np
import torchmetrics
# import os
# matplotlib.use("TkAgg")
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class M4(nn.Module):
    def __init__(self, n_output=12, n_channel=64):
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
        # plt.figure()
        # plt.hist(x.detach().cpu().view(-1))
        # plt.show()
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

if __name__ == '__main__':

    from aihwkit_lightning.nn.conversion import convert_to_analog
    from aihwkit_lightning.simulator.configs import (
        TorchInferenceRPUConfig,
        WeightClipType,
        WeightNoiseInjectionType,
    )

    from aihwkit_lightning.optim import AnalogOptimizer

    # from sklearn.metrics import confusion_matrix
    # import seaborn as sn
    # import pandas as pd
    # import matplotlib.pyplot as plt
    # import matplotlib.colors as mcolors
    # from torchlop import profile

    rpu_config = TorchInferenceRPUConfig()
    rpu_config.forward.inp_res = 2**8 - 2
    rpu_config.forward.out_noise = 0.075
    rpu_config.forward.out_noise_per_channel = True
    rpu_config.forward.out_bound = -1
    rpu_config.forward.out_res = -1

    rpu_config.clip.type = WeightClipType.LAYER_GAUSSIAN_PER_CHANNEL
    rpu_config.clip.sigma = 1.5

    rpu_config.modifier.noise_type = WeightNoiseInjectionType.ADD_NORMAL_PER_CHANNEL
    rpu_config.modifier.std_dev = 0.05

    rpu_config.mapping.max_input_size = -1

    rpu_config.pre_post.input_range.enable = True
    rpu_config.pre_post.input_range.init_from_data = 100
    rpu_config.pre_post.input_range.init_std_alpha = 3.0

    num_classes = 12
    batch_size = 10
    analog_train = True

    testset = torch.load("/mnt/c/Users/moham/OneDrive - University of Twente/Documenten/github/brainspy-tasks/bspytasks/temporal_kernels/GoogleSpeechCommands/dataset/dnpu_measurements/testset_kernel=8_12classes.pt", weights_only=False)

    # train_dataloader = torch.utils.data.DataLoader(
    #     dataset=trainset,
    #     batch_size = batch_size,
    #     shuffle= True,
    #     drop_last=False,
    # )

    val_dataloader = torch.utils.data.DataLoader(
        dataset=testset,
        batch_size = 10,
        shuffle= True,
        drop_last=False,
    )

    pytorch_model = M4(
        n_output=num_classes,
        n_channel=64
    )

    pytorch_model_cpu = pytorch_model.to('cpu')

    # macs, params, layer_infos = profile(pytorch_model, inputs=(torch.empty(1, 64, 1984),)) 
    # for layer_name, (layer_mac, layer_params) in layer_infos.items(): 
    #     print(f"{layer_name}: MACs = {layer_mac}, Parameters = {layer_params}")

    # best_model_path = "logs/dnpu_logs/analog_trained/version_27/checkpoints/epoch=384-step=32725.ckpt"
    # best_model_path = "logs/dnpu_logs/analog_trained/version_31/checkpoints/epoch=242-step=20655.ckpt"
    # best_model_path = "logs/dnpu_logs/analog_trained/version_32/checkpoints/epoch=162-step=13855.ckpt"
    # best_model_path = "logs/dnpu_logs/analog_trained/version_33/checkpoints/epoch=147-step=12580_kernel=8_decay=0-1.ckpt"
    # best_model_path = "logs/dnpu_logs/analog_trained/version_36/checkpoints/epoch=392-step=66417.ckpt"
    best_model_path = "lightning_logs/version_2/checkpoints/epoch=102-step=2266.ckpt"

    checkpoint = torch.load(best_model_path, weights_only=False)
    new_state_dict = {}
    for key, value in checkpoint['state_dict'].items():
        new_key = key.replace('model.', '')
        new_state_dict[new_key] = value
    
    model = convert_to_analog(pytorch_model, rpu_config, verbose=True)
    model.load_state_dict(new_state_dict)
    model.to(device)

    model.eval()
    accuracy_metric = torchmetrics.Accuracy(task='multiclass', num_classes=num_classes)
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for batch in val_dataloader:
            features, true_labels = batch
            logits = (model(features.to(device)).squeeze()).to(device)
            preds = torch.argmax(logits, dim=1)
            all_preds.append(preds)
            all_labels.append(true_labels)

    all_preds = torch.cat(all_preds)
    all_labels = torch.cat(all_labels)
    accuracy = accuracy_metric(all_preds.to('cpu'), all_labels.to('cpu'))
    print(f'Validation Accuracy: {accuracy.item()}')

    # print("plotting confusion matrix")


    # # plot confusion matrix
    # classes = ['Unknown', 'Down', 'Go', 'Left', 'No', 'Off', 'On', 'Right',
    #     'Stop', 'Up', 'Yes', 'House']
    # cf_matrix = confusion_matrix(all_labels.detach().cpu(), all_preds.detach().cpu())

    # # normalizing confusion matrix
    # cf_matrix = cf_matrix.astype('float') / cf_matrix.sum(axis=1)[:, np.newaxis]

    # df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None],
    #                     index = [i for i in classes],
    #                     columns = [i for i in classes])
    # plt.figure();sn.heatmap(df_cm, annot=False, fmt = '.2f', cmap='Blues', norm = mcolors.LogNorm())
    # accuracy_values = np.diag(df_cm)
    # for i in range(len(classes)):
    #     plt.text(i + 0.5, i + 0.5, f'{accuracy_values[i]:.2f}', 
    #             ha="center", va="center", color="black")

    # plt.ylabel('True label')
    # plt.xlabel('Predicted label')
    # plt.title('Confusion Matrix with Accuracy')
    # plt.show(block=True)

    # print("")
