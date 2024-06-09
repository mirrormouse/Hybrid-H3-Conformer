import os
import sys

repo_path = os.path.abspath("H3")
sys.path.append(repo_path)

import torch
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset
import tensorflow as tf
import numpy as np
from tqdm import tqdm
import librosa
import time
import traceback
from datetime import datetime
from jiwer import wer, cer
import json
import torch.optim as optim
from torch.optim.lr_scheduler import OneCycleLR
from torch.utils.data import Dataset, DataLoader
import threading
from model.conformer import H3_CausalConformer
from model.horizontal import Horizontal_CH4
import random
import torchaudio.transforms as T
from asrdata import ASRDataset, ASRDataset_Libri
import pickle
import torch.multiprocessing as mp


if __name__ == "__main__":

    mp.set_start_method('spawn', force=True)


    tf.config.threading.set_inter_op_parallelism_threads(2)

    ### parameters

    attn_layer =  [0,1]
    h3_head_dim = 8
    attn_dim = None
    N = 80
    batch_framesize = 100000
    num_worker = 3
    epoch_num = 50
    lr = 0.001

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    device_ids = [0]

    ### When training on LibriSpeech, Use sentencepiece

    ### CSJ mode
    id2char=json.load(open('tokenize/id2char_esp.json'))
    char2id=json.load(open('tokenize/char2id_esp.json'))
    Class = len(id2char)+1

    ### LibriSpeech Mode
    # spm_model = 'tokenize/spm_model.model'
    # sp = spm.SentencePieceProcessor(model_file=spm_model)
    # Class = sp.get_piece_size() + 1

    def makeloader(dataset, train_part=0.9, num_worker=4):
        train_size = int(train_part * len(dataset))
        val_size = len(dataset) - train_size
        train, val = torch.utils.data.random_split(dataset=dataset, lengths=[train_size, val_size], generator=torch.Generator().manual_seed(42))
        trainloader = torch.utils.data.DataLoader(train, batch_size=1,  shuffle=True,  drop_last=True, pin_memory = True, num_workers = num_worker)
        valloader = torch.utils.data.DataLoader(val, batch_size=1,  shuffle=True,  drop_last=True, pin_memory = True, num_workers = num_worker)
        return trainloader, valloader


    char_list = [c for c in char2id.keys()]

    ### Causal Conformer Mode
    model = H3_CausalConformer(input_dim = N, output_dim = Class, depth = 12, dim = 256, h3_dim_head = h3_head_dim, attn_layer_idx = attn_layer)
    ### Horizontal CH4 Mode
    # model = Horizontal_CH4(input_dim = N, output_dim = Class, depth = 12, dim = 256, h3_dim_head = h3_head_dim, attn_layer_idx = attn_layer, attn_dim = 32)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"学習パラメータ数: {total_params}")


    model = nn.DataParallel(model, device_ids=device_ids)
    model = model.to(device)




    datadir = "path/to/dataset"
    with open(f"{datadir}/train.pkl", "rb") as file:
        asrdata = pickle.load(file)
    with open(f"{datadir}/global_stat.pth", "rb") as file:
        global_stat = torch.load(file)

    global_mean, global_std = global_stat[0], global_stat[1]
    ### CSJ Mode
    dataset = ASRDataset(asrdata, global_mean, global_std)
    ### LibriSpeech Mode
    # dataset = ASRDataset_Libri(asrdata, global_mean, global_std, spm_model)


    num_steps = len(dataset)
    print(f"Dataset size:{len(dataset)}")


    trainloader, valloader = makeloader(dataset, train_part = 0.95, num_worker=num_worker)


    criterion = nn.CTCLoss(blank = 0, zero_infinity=True)
    optimizer = optim.AdamW(model.parameters(), lr=lr, amsgrad=True) 

    scheduler = OneCycleLR(optimizer, max_lr=lr, total_steps=epoch_num * num_steps, pct_start= 0.1 , anneal_strategy='cos')


    print("TRAIN START")
    training_start_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"学習開始時刻:{training_start_time}")


    time_masking = T.TimeMasking(time_mask_param=40)
    freq_masking = T.FrequencyMasking(freq_mask_param=40)
    

    loss_val_record = []
    loss_train_record = []

    start = time.time()

    try:
        for i in range(epoch_num):

            epoch = i
            loss_train = 0
            
            for j, (x, target, xl, tl) in tqdm(enumerate(trainloader)):

                x, target, xl, tl = x.squeeze(0), target.squeeze(0), xl.squeeze(0), tl.squeeze(0)
                optimizer.zero_grad()

                x = time_masking(x)
                x = freq_masking(x)
                x = x.to(device)
                xl = xl.to(device)
                output, xl = model(x, xl)

                output = torch.transpose(output, 0, 1)
                output = output.log_softmax(2)

                

                loss = criterion(output.to(device), target.to(device), xl.to(device), tl.to(device))


                loss_train += loss.item()


                loss.backward()
                optimizer.step()

                scheduler.step()

            loss_train = loss_train/len(trainloader)
            loss_train_record.append(loss_train)
            
            print("Epoch:", i, "Loss_Train:", loss_train)

            model.eval()


            loss_val = 0

            with torch.no_grad():
                for j, (x, target, xl, tl) in tqdm(enumerate(valloader)):

                    x, target, xl, tl = x.squeeze(0), target.squeeze(0), xl.squeeze(0), tl.squeeze(0)

                    x = x.to(device)
                    xl = xl.to(device)
                    output, xl = model(x, xl)

                    output = torch.transpose(output, 0, 1)
                    output = output.log_softmax(2)

                    loss = criterion(output.to(device), target.to(device), xl.to(device), tl.to(device))
                    loss_val += loss.item()

                loss_val = loss_val/len(valloader)
                loss_val_record.append(loss_val)

            print("Epoch:", i, "Loss_Train:", loss_val)
                
    except Exception as e:
        print(e)
        traceback.print_exc()

