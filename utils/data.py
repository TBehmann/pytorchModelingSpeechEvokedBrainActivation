import os
import pickle
import re
import numpy as np
from scipy import io
import torch
from torch.utils import data   # parent class for pytorch datasets

def load_matlab_matrix(sfile):
    """Load matrix in matlab's .mat format using numpy.io.loadmat"""
    return np.squeeze(io.loadmat(sfile)['data'])

def load_matlab_cell_array(sfile):
    """Load cell array in matlab's .mat format using numpy.io.loadmat"""
    return io.loadmat(sfile)['data']


def getChannels(dataset, analysis_chans=True):
    if dataset == 'YAU038':
        chans = [4, 5, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 24, 26, 28, 29, 30, 32]
        if analysis_chans:
            chans + [7, 15, 23]
    elif dataset == 'YAU039':
        chans = [3, 4, 5, 9, 10, 11, 12, 13, 14, 16, 18, 19, 20, 21, 24, 26, 27, 28, 29, 30, 32]
        if analysis_chans:
            chans + [7, 15, 23]
    else:
        chans = None
    return chans


def ecog_data_load(dataPath, dataset):
    """Load YAU038 data into d['X': X, 'y': y] format for pytorch"""
    d = dict()
    d['X'] = load_matlab_matrix( dataPath+os.path.sep+'features'+os.path.sep+dataset+'_audiospec.mat')
    d['y'] = load_matlab_matrix( dataPath+os.path.sep+'labels'+os.path.sep+dataset+'_labels.mat')
    d['y'][d['y'] == -1] = 0  # set '-1' labels to '0'
    cell_array = load_matlab_cell_array( dataPath+os.path.sep+'features'+os.path.sep+dataset+'_0000_spec_averaged.mat')
    d['features'] = cell_array[0] [0]
    d['f'] = cell_array[0] [1]
    d['features_shape'] = (d['features'].shape)
    d['features'] = np.reshape(d['features'], (d['features'].shape[0], d['features'].shape[1] * d['features'].shape[2]), order='F')
    d['envelope'] = load_matlab_cell_array( dataPath+os.path.sep+'envelopes'+os.path.sep+dataset+'_envelope_A-weighted')
    d['chans'] = getChannels(dataset, analysis_chans=True)

    return d


def load_dict_from_pickle(path):
    file = open(path, "rb")
    parameters = pickle.load(file)
    file.close()
    return parameters


def save_dict_to_pickle(dict, path):
    file = open(path + ".pkl", "wb")
    pickle.dump(dict, file)
    file.close()


def drop_extractor(str):
    matches=re.search('(?P<drop>\d_\d+)',str)
    return matches


def create_train_data_loader(dataset, batch_sz, shuffle, train_idx=None):
    if train_idx is None:
        train_loader = data.DataLoader(dataset=dataset, batch_size=batch_sz, shuffle=shuffle)
    else:
        train_loader = data.DataLoader(dataset=data.Subset(dataset, train_idx), batch_size=batch_sz, shuffle=shuffle)
    return train_loader


def create_validation_data_loader(dataset, val_idx):
        return data.DataLoader(dataset=data.Subset(dataset, val_idx), batch_size=val_idx.size)


class SimpleDataset(data.Dataset):
    def __init__(self,
                 dataPath='/home/tb/ecogstamo-release/data/pipelines/variant2_trimmed_labelWinStart_EnvLongWin_AudioSpecLeft',
                 dataset='YAU038'):
        d = ecog_data_load(dataPath, dataset)
        self.name = dataset
        self.X = torch.from_numpy(d['X']).float()
        self.y = torch.from_numpy(d['y']).float()
        self.features = torch.from_numpy(d['features']).float()
        self.features_shape = d['features_shape']
        self.envelope = torch.from_numpy(d['envelope'].T.squeeze()).float()
        self.f = d['f']
        self.chans = d['chans']

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx, :], self.y[idx], self.features[idx, :], self.envelope[idx]


