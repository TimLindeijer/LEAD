import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import warnings
import random
from data_provider.uea import (
    normalize_batch_ts,
    bandpass_filter_func,
    load_data_by_ids,
)

warnings.filterwarnings('ignore')

def get_id_list_caueeg(args, label_path, a=0.6, b=0.8):
    data_list = np.load(label_path)
    hc_list = list(data_list[np.where(data_list[:, 0] == 0)][:, 1])  # Healthy IDs
    mci_list = list(data_list[np.where(data_list[:, 0] == 1)][:, 1])  # MCI IDs
    dementia_list = list(data_list[np.where(data_list[:, 0] == 2)][:, 1])  # Dementia IDs
    
    if args.cross_val == 'fixed':
        random.seed(42)
    elif args.cross_val == 'mccv':
        random.seed(args.seed)
    elif args.cross_val == 'loso':
        all_ids = list(data_list[:, 1])
        hc_mci_dementia_list = sorted(hc_list + mci_list + dementia_list)
        test_ids = [hc_mci_dementia_list[(args.seed - 41) % len(hc_mci_dementia_list)]]
        train_ids = [id for id in hc_mci_dementia_list if id not in test_ids]
        random.seed(args.seed)
        random.shuffle(train_ids)
        val_ids = train_ids[int(0.9 * len(train_ids)):]
        return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)
    else:
        raise ValueError('Invalid cross_val. Please use fixed, mccv, or loso.')
    
    random.shuffle(hc_list)
    random.shuffle(mci_list)
    random.shuffle(dementia_list)
    
    all_ids = list(data_list[:, 1])
    train_ids = (hc_list[:int(a * len(hc_list))] +
                 mci_list[:int(a * len(mci_list))] +
                 dementia_list[:int(a * len(dementia_list))])
    val_ids = (hc_list[int(a * len(hc_list)):int(b * len(hc_list))] +
               mci_list[int(a * len(mci_list)):int(b * len(mci_list))] +
               dementia_list[int(a * len(dementia_list)):int(b * len(dementia_list))])
    test_ids = (hc_list[int(b * len(hc_list)):] +
                mci_list[int(b * len(mci_list)):] +
                dementia_list[int(b * len(dementia_list)):])
    
    return sorted(all_ids), sorted(train_ids), sorted(val_ids), sorted(test_ids)

class CAUEEGLoader(Dataset):
    def __init__(self, args, root_path, flag=None):
        self.no_normalize = args.no_normalize
        self.root_path = root_path
        self.data_path = os.path.join(root_path, 'Feature/')
        self.label_path = os.path.join(root_path, 'Label/label.npy')
        
        a, b = 0.6, 0.8
        self.all_ids, self.train_ids, self.val_ids, self.test_ids = get_id_list_caueeg(args, self.label_path, a, b)
        if flag == 'TRAIN':
            ids = self.train_ids
            print('train ids:', ids)
        elif flag == 'VAL':
            ids = self.val_ids
            print('val ids:', ids)
        elif flag == 'TEST':
            ids = self.test_ids
            print('test ids:', ids)
        elif flag == 'PRETRAIN':
            ids = self.all_ids
            print('all ids:', ids)
        else:
            raise ValueError('Invalid flag. Please use TRAIN, VAL, TEST, or PRETRAIN.')
        
        self.X, self.y = load_data_by_ids(self.data_path, self.label_path, ids)
        self.y[:, 0] = np.where(self.y[:, 0] == 2, 1, self.y[:, 0])  # Merge Dementia into MCI class
        self.X = bandpass_filter_func(self.X, fs=args.sampling_rate, lowcut=args.low_cut, highcut=args.high_cut)
        self.X = normalize_batch_ts(self.X)
        self.max_seq_len = self.X.shape[1]

    def __getitem__(self, index):
        return torch.from_numpy(self.X[index]), torch.from_numpy(np.asarray(self.y[index]))

    def __len__(self):
        return len(self.y)