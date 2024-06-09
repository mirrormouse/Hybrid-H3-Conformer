import json
from glob import glob
import os
import librosa
import numpy as np
from tqdm import tqdm
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from torch.nn.utils.rnn import pad_sequence
import torchaudio
import gc
import pickle
import torch
from torch.utils.data import Dataset
import threading

from concurrent.futures import ThreadPoolExecutor
import torch.multiprocessing as mp
import sentencepiece as spm



class ASRDataset(Dataset):
    def __init__(self, data_list, global_mean, global_std, device="cpu"):
        self.data_list = data_list
        self.device = device
        self.global_mean = global_mean.to(self.device)
        self.global_std = global_std.to(self.device)
        self.char2id = char2id=json.load(open('tokenize/char2id_esp.json'))
        
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        tensor_files, labels = self.data_list[idx]
        spec = [self.normalize(torch.load((file))) for file in tensor_files]

        spec_length = torch.tensor([s.shape[0] for s in spec], dtype=torch.int)

        max_length = max(spec_length)
        spec = [self.pad_tensor(s, max_length) for s in spec]
        spec = torch.stack(spec)


        target = [self.process_label(label) for label in labels]
        target_length = torch.tensor([len(t) for t in target], dtype=torch.int)

        # Pad target to have equal lengths
        max_length = max(target_length)
        target = [self.pad_tensor(t.to(self.device), max_length) for t in target]
        target = torch.stack(target)

        target = target.to(torch.int)
        return (spec, target, spec_length, target_length)

    def load_and_normalize(self, file, spec, i):
        # Load tensor data and normalize it
        spec[i] = self.normalize(torch.load(file))


    def normalize(self, tensor):
        return (tensor.to(self.device) - self.global_mean) / self.global_std

    def process_label(self,label):
        # Transform text label into ASCII tensor. Modify this part if you have a specific way to process labels.
        return torch.tensor([self.char2id[c] for c in label])

    def pad_tensor(self, tensor, length):
        pad_size = list(tensor.shape)
        pad_size[0] = length - tensor.size(0)
        return torch.cat([tensor, torch.zeros(*pad_size).to(self.device)], dim=0)





class ASRDataset_Libri(Dataset):
    def __init__(self, data_list, global_mean, global_std, spm_model, device="cpu"):
        self.data_list = data_list
        self.device = device
        self.global_mean = global_mean.to(self.device)
        self.global_std = global_std.to(self.device)
        self.sp = spm.SentencePieceProcessor(model_file=spm_model)
        
    def __len__(self):
        return len(self.data_list)
    def __getitem__(self, idx):
        tensor_files, labels = self.data_list[idx]

        spec = [self.normalize(torch.load(file)) for file in tensor_files]

        spec_length = torch.tensor([s.shape[0] for s in spec], dtype=torch.int)

        max_length = max(spec_length)
        spec = [self.pad_tensor(s, max_length) for s in spec]
        spec = torch.stack(spec)


        target = [self.process_label(label) for label in labels]

        target_length = torch.tensor([len(t) for t in target], dtype=torch.int)

        # Pad target to have equal lengths
        max_length = max(target_length)
        target = [self.pad_tensor(t.to(self.device), max_length) for t in target]
        target = torch.stack(target)

        target = target.to(torch.int)
        return (spec, target, spec_length, target_length)

    def load_and_normalize(self, file, spec, i):
        # Load tensor data and normalize it
        spec[i] = self.normalize(torch.load(file))


    def normalize(self, tensor):
        return (tensor.to(self.device) - self.global_mean) / self.global_std

    def process_label(self,label):
        # Transform text label into ASCII tensor. Modify this part if you have a specific way to process labels.
        token = self.sp.encode_as_ids(label)
        return torch.tensor([data+1 for data in token])

    def pad_tensor(self, tensor, length):
        pad_size = list(tensor.shape)
        pad_size[0] = length - tensor.size(0)
        return torch.cat([tensor, torch.zeros(*pad_size).to(self.device)], dim=0)
