import abc
import os
import pandas as pd
import torch
from sklearn.metrics import r2_score

import utils.data
from matplotlib import pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
import numpy as np
from scipy.io import savemat
from torchview import draw_graph


def evaluate_DNN(name, path, save_path=None, data_path=None, meta_data_files=[], scorers=[], plotters=[]):
    results_csv = []
    results_mat = dict()


    if len(meta_data_files) == 0:
        meta_data_files = [i for i in os.listdir(path) if i.endswith(".pkl")]

    test_datasets = dict()

    for index, meta_data_file in enumerate(meta_data_files):
        meta_data_path = path + os.sep + meta_data_file

        meta_data = utils.data.load_dict_from_pickle(meta_data_path)

        if meta_data.get('dataset') not in test_datasets:
            if data_path is None:
                test_datasets.update({meta_data.get('dataset'): utils.data.SimpleDataset(dataset=meta_data.get('dataset'))})
            else:
                test_datasets.update({meta_data.get('dataset'): utils.data.SimpleDataset(dataPath=data_path, dataset=meta_data.get('dataset'))})


        model_scores_mat = []
        fold_order = []
        scorer_order = []

        for fold, model_name in meta_data.get('model_relative_paths').items():
            model_class = meta_data.get('model_class')
            model_class.model_creation(meta_data)
            model = model_class.model
            model.load_state_dict(torch.load(path + os.sep + model_name))
            model.eval()

            fold_scores_mat = []
            fold_order.append(fold)

            model_scores_csv = dict()
            model_scores_csv.update({'name': meta_data_file})
            model_scores_csv.update({'fold': fold})
            for scorer in scorers:
                scorer_order.append(scorer.name)
                scores = scorer.score(
                    model,
                    test_datasets.get(meta_data.get('dataset')),
                    meta_data.get('idx_val')[fold]
                )

                if hasattr(scores, '__len__') and (not isinstance(scores, str)):
                    fold_scores_mat.append(scores)
                else:
                    model_scores_csv.update({scorer.name: scores})

            model_scores_mat.append(fold_scores_mat)
            results_csv.append(model_scores_csv)
            for plotter in plotters:
                plotter.plot(
                    model,
                    test_datasets.get(meta_data.get('dataset')),
                    meta_data.get('idx_val')[fold],
                    save_path
                )

        model_dict_mat = dict({'scores': model_scores_mat,
                               'fold_order': fold_order,
                               'scorer_order': scorer_order,
                               'freqs':test_datasets.get(meta_data.get('dataset')).f,
                               'chans':test_datasets.get(meta_data.get('dataset')).chans})
        results_mat.update({meta_data_file: model_dict_mat})

    if scorers:
        for meta_data_file_name, model_dict in results_mat.items():
            savemat(save_path + os.sep + name + '_' + meta_data_file_name + '.mat', model_dict)

        df = pd.DataFrame(results_csv)
        means = df.groupby('name').mean().reset_index()
        means['fold'] = 'mean'
        std = df.groupby('name').std().reset_index()
        std['fold'] = 'std'
        result_df = df.append(means, ignore_index=True).append(std, ignore_index=True)
        result_df.sort_values(by=['name','fold']).to_csv(meta_data_path + os.sep + name + '.csv', sep='\t', encoding='utf-8')

def plot_roc_cur(target, pred):
    fper, tper, _ = roc_curve(target, pred)
    plt.plot(fper, tper, color='orange', label='ROC')
    plt.plot([0, 1], [0, 1], color='darkblue', linestyle='--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend()
    plt.show()

def plot_pred_true_label_comp(pred, label, start, end, sample_freq = 200, binary = False): #TODO 2000?
    x = np.arange(0, pred.size)

    if binary:
        pred = convert_sigmoid_output_to_binary_output(pred)

    f = plt.figure()
    f.set_figwidth(12.8)
    plt.plot(x[start*sample_freq:end*sample_freq] / sample_freq, pred[start*sample_freq:end*sample_freq], color='orange', label='Prediction')
    plt.plot(x[start*sample_freq:end*sample_freq] / sample_freq, label[start*sample_freq:end*sample_freq], color='blue', label='True Class Label')
    plt.xlabel('Time [s]')
    plt.ylabel('Confidence for Label Speech')
    plt.title('DNN raw predictions')
    plt.legend()
    plt.show()

def convert_sigmoid_output_to_binary_output(sigmoid_output):
    return (sigmoid_output > 0.5).astype(int)

def generate_model_prediction(X, model):
    results = model(X)
    output_tuple = ()
    if  isinstance(results, tuple):
        for result in results:
            output_tuple = output_tuple + (result.detach().numpy().squeeze(),)
        return output_tuple
    else:
        return results.detach().numpy().squeeze()

class Scorer(metaclass = abc.ABCMeta):

    def __init__(self, name=''):
        self.name=name

    @abc.abstractmethod
    def score(self, model, data, idx):
        """Scores the given data defined by the given indices"""
        return

class Plotter(metaclass = abc.ABCMeta):

    def __init__(self, name=''):
        self.name=name

    @abc.abstractmethod
    def plot(self, model, data, idx, savePath):
        """Plots the given data defined by the given indices"""

class F1_Scorer(Scorer):

    def __init__(self):
        super().__init__(type(self).__name__)

    def score(self, model, data, idx):
        X, y, *rest = data[idx]
        pred = generate_model_prediction(X, model)
        if isinstance(pred, tuple):
            pred = pred[0]
        f1 = f1_score(y, convert_sigmoid_output_to_binary_output(pred))
        return f1

class ROC_AUC_Scorer(Scorer):

    def __init__(self):
        super().__init__(type(self).__name__)

    def score(self, model, data, idx):
        X, y, *rest = data[idx]
        pred = generate_model_prediction(X, model)
        if isinstance(pred, tuple):
            pred = pred[0]
        ROC=roc_auc_score(y,pred)
        return ROC

class R2_Scorer(Scorer):

    def __init__(self):
        super().__init__(type(self).__name__)

    def score(self, model, data, idx):
        X, _, _, envelope, *rest = data[idx]
        _, envelope_pred = generate_model_prediction(X, model)
        R2 = r2_score(envelope,envelope_pred)
        return R2

class Pred_True_Label_Comparision_Plotter(Plotter):

    def __init__(self, start, stop, sample_freq = 200, binary = False):
        super().__init__(type(self).__name__)
        self.start = start
        self.stop = stop
        self.sample_freq = sample_freq
        self.binary = binary

    def plot(self, model, data, idx, savePath):
        X, y, *rest = data[idx]
        pred = generate_model_prediction(X, model)
        if isinstance(pred, tuple):
            pred = pred[0]
        plot_pred_true_label_comp(pred, y, self.start, self.stop, binary=False)


class ROC_Curve_Plotter(Plotter):

    def __init__(self):
        super().__init__(type(self).__name__)

    def plot(self, model, data, idx, savePath):
        X, y, *rest = data[idx]
        pred = generate_model_prediction(X, model)
        if isinstance(pred, tuple):
            pred = pred[0]
        plot_roc_cur(y,pred)

class Model_Plotter(Plotter):

    def __init__(self):
        super().__init__(type(self).__name__)

    def plot(self, model, data, idx, savePath=None):
        X, *rest = data[0]
        input_size = list(X.shape)
        input_size.insert(0,1)
        if savePath is None:
            draw_graph(model, graph_name=type(model).__name__, input_size= input_size,
                       expand_nested=True).visual_graph.render(format='svg')
        else:
            draw_graph(model, graph_name=type(model).__name__, input_size= input_size,
                       expand_nested=True).visual_graph.render(directory=savePath, format='svg')

def test(path):
    a = utils.data.load_dict_from_pickle(path+'.pkl')
    d = dict()
    for i in range(5):
        d.update({i: path+'_fold_'+str(i)+'.pth'})
    a.update({'model_paths': d})
    utils.data.save_dict_to_pickle(a,path)
    return a

def temp(path):
    a = utils.data.load_dict_from_pickle(path+'.pkl')
    a.update({'model_paths': {0: path + 'no_folds.pth'}})
    utils.data.save_dict_to_pickle(a,path)
    return a

def test2(path):
    a = utils.data.load_dict_from_pickle(path+'.pkl')
    d = dict()
    for i, model_path in  a.get('model_paths').items():
        head, tail = os.path.split(model_path)
        d.update({i: tail})
    a.update({'model_relative_paths': d})
    utils.data.save_dict_to_pickle(a,path)
    return a