import os
import random
import numpy as np

from scipy.signal import stft
from scipy.io import wavfile
from scipy.signal import resample

import torch
import torch.nn as nn

from audiomentations import AddGaussianNoise, PitchShift, TimeMask, SpecFrequencyMask, Normalize, AddGaussianSNR

# converts stft into a tensor
def tensorify(Zxx):
    input = torch.tensor(Zxx)
    # Add an extra dimension to represent the channels
    input = input.unsqueeze(0)
    # Add an extra dimension to represent the batch size
    input = input.unsqueeze(0)
    return input

# augments a signal and returns the augmented signal's stft
def augment(data, sr):
    data = data.astype(np.float32)
    data = resample(data, int(len(data)/sr*8000))
    
    # add noise
    data = AddGaussianSNR(min_snr_db=5, max_snr_db=30, p=0.9)(data, sr)
    
    # time mask
    masks = random.randint(5, 8)
    for i in range(masks):
        data = TimeMask(p=1, max_band_part=0.03)(data, sr)

    f, t, Zxx = stft(data, 8000, nfft=1024)
    
    # frequency mask
    masks = random.randint(5, 8)
    for i in range(masks):
        Zxx = SpecFrequencyMask(p=1, max_mask_fraction=0.03)(Zxx)
    Zxx = np.abs(Zxx)
    
    # shift the signal up/down
    roll = random.randint(-70, 70)
    Zxx = np.roll(Zxx, roll, axis=0)
    
    # normalisation
    Zxx = Zxx + 1
    Zxx = np.log(Zxx)
    Zxx = Zxx/np.median(Zxx)
        
    return f, t, Zxx
    
# code for NT-Xent Loss function
def loss_function(a, b, temperature=0.05):
    a_norm = torch.norm(a, dim=1).reshape(-1, 1)
    a_cap = torch.div(a, a_norm)
    b_norm = torch.norm(b, dim=1).reshape(-1, 1)
    b_cap = torch.div(b, b_norm)
    a_cap_b_cap = torch.cat([a_cap, b_cap], dim=0)
    a_cap_b_cap_transpose = torch.t(a_cap_b_cap)
    b_cap_a_cap = torch.cat([b_cap, a_cap], dim=0)
    sim = torch.mm(a_cap_b_cap, a_cap_b_cap_transpose)
    sim_by_tau = torch.div(sim, temperature)
    exp_sim_by_tau = torch.exp(sim_by_tau)
    sum_of_rows = torch.sum(exp_sim_by_tau, dim=1)
    exp_sim_by_tau_diag = torch.diag(exp_sim_by_tau)
    numerators = torch.exp(torch.div(torch.nn.CosineSimilarity()(a_cap_b_cap, b_cap_a_cap), temperature))
    denominators = sum_of_rows - exp_sim_by_tau_diag
    num_by_den = torch.div(numerators, denominators)
    neglog_num_by_den = -torch.log(num_by_den)
    return torch.mean(neglog_num_by_den)

# Basic block for ResNet
class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        # Shortcut connection
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += self.shortcut(identity)
        out = self.relu(out)

        return out

# ResNet model
class ResNet(nn.Module):
    def __init__(self, embedding_size):
        super(ResNet, self).__init__()
        self.embedding_size = embedding_size

        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.embedding_size)

        # Initialize the weights
        self._initialize_weights()

    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = []
        layers.append(ResBlock(in_channels, out_channels, stride=stride))
        for _ in range(1, num_blocks):
            layers.append(ResBlock(out_channels, out_channels))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.maxpool(out)

        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)

        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out