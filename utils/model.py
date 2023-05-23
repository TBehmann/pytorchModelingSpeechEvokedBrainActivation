import abc
import os
import torch
import torch.nn as nn

def set_cuda_device(id=0):
    """Set CUDA GPU device to integer number, starting from PCI bus id 0 (default)."""
    if torch.cuda.is_available():
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        os.environ["CUDA_VISIBLE_DEVICES"] = str(id)
        device = torch.device('cuda:'+str(id))
    else:
        device = torch.device('cpu')
    print('Using PyTorch version:', torch.__version__, ' Device:', device)
    return device


def init_weights(module):
    if type(module) == nn.Linear or type(module) == MaxNormConstrainedLinear:
        torch.nn.init.xavier_uniform_(module.weight)
        module.bias.data.fill_(0.1)


def max_norm(weights, max_val=3, eps=1e-8):
    with torch.no_grad():
        norm = weights.norm(2, dim=0, keepdim=True)
        desired = torch.clamp(norm, 0, max_val)
        weights *= (desired / (eps + norm))  # this implementation scales weights with l2 norm lower than max_val
        # slightly but it mimics the keras implementation
        # ("https://github.com/keras-team/keras/blob/master/keras/constraints.py") which was the main goal


def _get_activation(name, dict):
    def hook(model, input, output):
        dict[name] = output.detach()
    return hook


def _register_hooks_to_module_ReLU(model, dict):
    for name, module in model.named_modules():
        if type(module) == nn.ReLU:
            module.register_forward_hook(_get_activation(name, dict))

def _register_hooks_to_module_Sigmoid(model, dict):
    for name, module in model.named_modules():
        if type(module) == nn.Sigmoid:
            module.register_forward_hook(_get_activation(name, dict))


def get_layer_activations(model, input):
    activations = {}

    _register_hooks_to_module_ReLU(model, activations)

    model(input)

    return activations


class MaxNormConstrainedLinear(nn.Linear):
    def forward(self, input):
        max_norm(self.weight)
        return nn.functional.linear(input, self.weight, self.bias)

class Model(metaclass = abc.ABCMeta):

    def __init__(self, architecture_class):
        self.architecture_class = architecture_class
        self.model = None
        self.meta_data = dict()

    @abc.abstractmethod
    def model_creation(self, input):
        """Creates the model with the given parameters"""
        return

    @abc.abstractmethod
    def train_DNN(self, input):
        """Trains the DNN with the given parameters"""
        return