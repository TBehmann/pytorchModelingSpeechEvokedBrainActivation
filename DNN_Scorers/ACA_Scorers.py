from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

import utils.evaluation
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy import stats

class ACA_Layer_Scorer(utils.evaluation.Scorer):

    def __init__(self):
        super().__init__(type(self).__name__)

    def _score_features(self, features, layer_activations, features_shape):
        ''' Returns a length (n_hidden_layers+1) list of (n_freqs, n_chans) ndarrays
        '''

        n_samples, n_freqs, n_chans = features_shape
        layer_scores = []

        for layer_activation in layer_activations:
            if layer_activation.shape[1] > 0:
                # Layer has active units
                feature_scores = []

                for feature in features.numpy().T:
                    # Regress activations on ECoG feature
                    lr_model = LinearRegression()
                    lr_model.fit(layer_activation, feature)

                    # Calculate feature score
                    r2 = stats.pearsonr(lr_model.predict(layer_activation), feature)[0] ** 2
                    feature_scores.append(r2)

                feature_scores = np.asarray(feature_scores).reshape((n_freqs, n_chans), order='F')

            elif layer_activation.shape[1] == 0:
                # Layer has no active units, can't fit the model.
                # Set correlation to NaN for all features.
                feature_scores = np.zeros((n_freqs, n_chans))
                feature_scores.fill(np.nan)

            layer_scores.append(feature_scores)

        return layer_scores


    def score(self, model, data, *args):
        X, y, ecog_features, *rest = data[:]
        # Get DNN layer activations
        layer_activations = _get_dnn_activations(model, X)
           # Get (per layer) feature scores
        layer_scores= self._score_features(ecog_features, layer_activations, (data.features_shape))
        return layer_scores

class ACA_Unit_Scorer(utils.evaluation.Scorer):

    def __init__(self):
        super().__init__(type(self).__name__)

    def _score_features(self, features, layer_activations, features_shape):
        ''' Returns a length (n_hidden_layers+1) list of (n_freqs, n_chans) ndarrays
        '''
        n_samples, n_freqs, n_chans =  features_shape
        layer_scores = []
        kmeans = []
        for layer_activation in layer_activations:
            if layer_activation.shape[1] > 0:
                # Layer has active units
                unit_scores = []
                for unit in layer_activation.T:
                    feature_scores = []
                    unit = unit.reshape(-1, 1)

                    for feature in features.numpy().T:
                        lr_model = LinearRegression()
                        lr_model.fit(unit, feature)
                        r2 = stats.pearsonr(lr_model.predict(unit), feature)[0] ** 2
                        feature_scores.append(r2)
                    feature_scores = np.asarray(feature_scores).reshape((n_freqs, n_chans), order='F')
                    unit_scores.append(np.atleast_3d(feature_scores))
            elif layer_activation.shape[1] == 0:
                # Layer has no active units, can't fit the model.
                # Set correlation to NaN for all features.
                unit_scores = np.zeros((n_freqs, n_chans, (len(layer_activation), 2)))
                unit_scores.fill(np.nan)

            unit_scores_mat = np.concatenate(unit_scores, axis=2)
            unit_scores_2D_mat = np.reshape(unit_scores_mat, (
            unit_scores_mat.shape[0] * unit_scores_mat.shape[1], unit_scores_mat.shape[2])).T
            if unit_scores_2D_mat.shape[0] > 1:
                silhouette_scores = []
                kMeans_temp = []
                for n in range(2, 11):
                    kInstance = KMeans(n_clusters=n).fit(unit_scores_2D_mat)
                    silhouette_scores.append(silhouette_score(unit_scores_2D_mat, kInstance.labels_))
                    kMeans_temp.append(kInstance)
                index_max: int = max(range(len(silhouette_scores)), key=silhouette_scores.__getitem__)
                optimal_kInstance = kMeans_temp[index_max]
                kmeans.append(optimal_kInstance.labels_)
            layer_scores.append(unit_scores)
        return layer_scores, kmeans

    def score(self, model, data, *args):
        X, y, ecog_features, *rest = data[:]
        # Get DNN layer activations
        layer_activations = _get_dnn_activations(model, X)
        # Get (per layer) feature scores
        layer_scores, kmeans = self._score_features(ecog_features, layer_activations, data.features_shape)
        return (layer_scores, kmeans)

def _get_dnn_activations(model, X):
    ''' Returns a length (n_hidden_layers+1) list of (n_samples, n_active_units) ndarrays
    '''
    # Get all layer activations elicited by the test data
    layer_activations = utils.model.get_layer_activations(model, X)
    # Remove inactive units, represented by all-zero columns
    layer_activations = [layer_activation.numpy()[:, layer_activation.bool().any(axis=0).numpy()] for layer_activation in
                         layer_activations.values()]
    return layer_activations