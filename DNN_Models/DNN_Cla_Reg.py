import os
import threading
import time
from sklearn import model_selection
import torch
from torch import nn
import utils.model

class cla_reg_model(utils.model.Model):

    def __init__(self, architecture_class):
        super().__init__(architecture_class)

    def model_creation(self, kwargs):
        defaultKwargs = {'n_features': 2070, 'drop_visible': 0.2, 'drop_hidden': 0.5, 'init_weights': True}
        kwargs = {**defaultKwargs, **kwargs}

        self.meta_data.update({'n_features': kwargs.get('n_features'),
                               'drop_hidden': kwargs.get('drop_hidden'),
                               'drop_visible': kwargs.get('drop_visible'),
                               'init_weights': kwargs.get('init_weights}')})

        self.model = self.architecture_class(kwargs)

        if kwargs.get('init_weights}'):
            self.model.apply(utils.model.init_weights)

    def _train(self, train_loader, optimizer, criterionCla, criterionReg, device, epoch, log_interval=200, verbose=True):
        model = self.model.to(device)
        # Set model to training mode
        model.train()

        # Loop over each batch from the training set
        for batch_idx, (data, targetCla, _, targetReg, *rest) in enumerate(train_loader):
            # Push data to GPU if needed (depends on dataloader structure)
            data = data.to(device)
            targetCla = targetCla.to(device)
            targetReg = targetReg.to(device)

            # Zero gradient buffers
            optimizer.zero_grad()

            # Pass data through the network
            outputCla, outputReg = model(data)

            # Calculate loss
            lossCla = criterionCla(torch.squeeze(outputCla), targetCla)
            lossReg = criterionReg(torch.squeeze(outputReg), targetReg)

            # Backpropagate
            (lossCla + lossReg).backward()

            # Update weights
            optimizer.step()

            if verbose == True:
                if batch_idx % log_interval == 0:
                    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLossCla: {:.6f}\tLossReg: {:.6f}'.format(
                        epoch, batch_idx * len(data), len(train_loader.dataset),
                               100. * batch_idx / len(train_loader), lossCla.data.item(), lossReg.data.item()))

    def train_DNN(self, dataset, sema=threading.BoundedSemaphore(1),
                  save_path = '/home/tb/Documents/DNN', name ='DNN_Cla_Reg_Default', optim_class='Adam', lr=0.001,
                  batch_sz=32, epochs=1000, device_id=0, momentum=0.5, folds=1, shuffle=True, verbose=False):
        self.meta_data.update({'start_time': time.time()})

        device = utils.model.set_cuda_device(device_id)


        ##################################################################################

        if optim_class=='SGD':
            optimizer = torch.optim.SGD(self.model.parameters(), lr, momentum)     # Create Optimizer
        else:
            optimizer = torch.optim.Adam(self.model.parameters(), lr)
        criterionCla = nn.BCELoss()       # And loss function
        criterionReg = nn.MSELoss()

        if verbose:
            print(self.model)    # Print Model Architectures

        model_names = dict()
        if folds > 1:
            k_fold = model_selection.KFold(n_splits=folds, shuffle=False)
            cv_idx_train_list = []
            cv_idx_val_list = []
            # scores = []
            for fold, (train_idx, val_idx) in enumerate(k_fold.split(dataset)):

                train_loader = utils.data.create_train_data_loader(dataset, batch_sz, shuffle, train_idx=train_idx)

                self.model.apply(utils.model.init_weights)

                # Actual training loop
                with sema:
                    for epoch in range(1, epochs + 1):
                        self._train(train_loader, optimizer, criterionCla, criterionReg, device, epoch, verbose=verbose)

                model_name = name + '_fold_' + str(fold) + '.pth'
                model_path = save_path + os.sep + model_name
                model_names.update({fold: model_name})
                torch.save(self.model.state_dict(), model_path)
                cv_idx_train_list.append(train_idx)
                cv_idx_val_list.append(val_idx)

            self.meta_data.update({'idx_train': cv_idx_train_list,
                                   'idx_val': cv_idx_val_list})

        else:
            train_loader = utils.data.create_train_data_loader(dataset, batch_sz, shuffle)

            self.model.apply(utils.model.init_weights)

            # Actual training loop
            with sema:
                for epoch in range(1, epochs + 1):
                    self._train(train_loader, optimizer, criterionCla, criterionReg, device, epoch, verbose=verbose)

            model_name = name + '_no_folds' + '.pth'
            model_path = save_path + os.sep + model_name
            model_names.update({0: model_name})
            torch.save(self.model.state_dict(), model_path)

            self.meta_data.update({'idx_train': [range(0, len(dataset))],
                                   'idx_val': [range(0, len(dataset))]})

        self.meta_data.update({'dataset': dataset.name,
                               'lr': lr,
                               'batch_sz': batch_sz,
                               'epochs': epochs,
                               'optimizer_type': optim_class,
                               'momentum': momentum if optim_class == 'Adam' else None,
                               'shuffle': shuffle,
                               'folds': folds,
                               'model_class': cla_reg_model(self.architecture_class),
                               'model_relative_paths': model_names})
        utils.data.save_dict_to_pickle(self.meta_data, save_path + os.sep + name)


def default_forward(self, x):
  x = self.envelope_extraction_layers(x)
  envelope = self.envelope_layer(x)
  x = torch.cat((x, envelope), dim=1)
  output = self.higher_processing_layers(x)
  return output, envelope

class EnvelopeLayerAfterFirstLinearLayerArchitecture(nn.Module):
    def __init__(self, kwargs):
      super(EnvelopeLayerAfterFirstLinearLayerArchitecture, self).__init__()
      self.envelope_extraction_layers = nn.Sequential(
          nn.Dropout(kwargs.get('drop_visible')),
          utils.model.MaxNormConstrainedLinear(kwargs.get('n_features'), int(kwargs.get('n_features') / 4)),
          nn.ReLU(),
          nn.Dropout(kwargs.get('drop_hidden'))
      )

      self.envelope_layer = nn.Sequential(
          utils.model.MaxNormConstrainedLinear(int(kwargs.get('n_features') / 4), 1),
          nn.ReLU()
      )

      self.higher_processing_layers = nn.Sequential(
          utils.model.MaxNormConstrainedLinear(int(kwargs.get('n_features') / 4)+1, int(kwargs.get('n_features') / 16)),
          nn.ReLU(),
          nn.Dropout(kwargs.get('drop_hidden')),
          utils.model.MaxNormConstrainedLinear(int(kwargs.get('n_features') / 16), int(kwargs.get('n_features') / 64)),
          nn.ReLU(),
          nn.Dropout(kwargs.get('drop_hidden')),
          nn.Linear(int(kwargs.get('n_features') / 64), 1),
          nn.Sigmoid()
      )

    # x represents our data
    def forward(self, x):
      return default_forward(self, x)

class EnvelopeLayerAfterSecondLinearLayerArchitecture(nn.Module):
    def __init__(self, kwargs):
      super(EnvelopeLayerAfterSecondLinearLayerArchitecture, self).__init__()
      self.envelope_extraction_layers = nn.Sequential(
          nn.Dropout(kwargs.get('drop_visible')),
          utils.model.MaxNormConstrainedLinear(kwargs.get('n_features'), int(kwargs.get('n_features') / 4)),
          nn.ReLU(),
          nn.Dropout(kwargs.get('drop_hidden')),
          utils.model.MaxNormConstrainedLinear(int(kwargs.get('n_features') / 4) , int(kwargs.get('n_features') / 16)),
          nn.ReLU(),
          nn.Dropout(kwargs.get('drop_hidden')),
      )

      self.envelope_layer = nn.Sequential(
          utils.model.MaxNormConstrainedLinear(int(kwargs.get('n_features') / 16), 1),
          nn.ReLU()
      )

      self.higher_processing_layers = nn.Sequential(
          utils.model.MaxNormConstrainedLinear(int(kwargs.get('n_features') / 16)+1, int(kwargs.get('n_features') / 64)),
          nn.ReLU(),
          nn.Dropout(kwargs.get('drop_hidden')),
          nn.Linear(int(kwargs.get('n_features') / 64), 1),
          nn.Sigmoid()
      )

    # x represents our data
    def forward(self, x):
      return default_forward(self, x)

class EnlargedEnvelopeLayerAfterFirstLinearLayerArchitecture(nn.Module):
    def __init__(self, kwargs):
        super(EnlargedEnvelopeLayerAfterFirstLinearLayerArchitecture, self).__init__()
        self.envelope_extraction_layers = nn.Sequential(
            nn.Dropout(kwargs.get('drop_visible')),
            utils.model.MaxNormConstrainedLinear(kwargs.get('n_features'), int(kwargs.get('n_features') / 4)),
            nn.ReLU(),
            nn.Dropout(kwargs.get('drop_hidden'))
        )
        self.envelope_layer = nn.Sequential(
            utils.model.MaxNormConstrainedLinear(int(kwargs.get('n_features') / 4),
                                                 int(kwargs.get('n_features') / 16)),
            nn.ReLU(),
            nn.Dropout(kwargs.get('drop_hidden')),
            utils.model.MaxNormConstrainedLinear(int(kwargs.get('n_features') / 16),
                                                 1),
            nn.ReLU()
        )
        self.higher_processing_layers = nn.Sequential(
            utils.model.MaxNormConstrainedLinear(int(kwargs.get('n_features') / 4) + 1,
                                                 int(kwargs.get('n_features') / 16)),
            nn.ReLU(),
            nn.Dropout(kwargs.get('drop_hidden')),
            utils.model.MaxNormConstrainedLinear(int(kwargs.get('n_features') / 16),
                                                 int(kwargs.get('n_features') / 64)),
            nn.ReLU(),
            nn.Dropout(kwargs.get('drop_hidden')),
            nn.Linear(int(kwargs.get('n_features') / 64), 1),
            nn.Sigmoid()
        )
    # x represents our data
    def forward(self, x):
        return default_forward(self, x)