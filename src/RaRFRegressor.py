import numpy as np
from scipy.spatial.distance import euclidean, jaccard
from sklearn.ensemble import RandomForestRegressor
from tqdm import tqdm 
from joblib import Parallel, delayed
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.feature_selection import VarianceThreshold
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings

import sys
sys.path.append('../src/')
import RaRFRegressor
import utils

plt.rcParams['font.size'] = 14
COLORA = '#027F80'
COLORB = '#B2E5FC'

class RaRFRegressor:
    def __init__(self, *, radius=1, metric='jaccard'):
        self.radius = radius
        if metric=='euclidean':
            self.metric = euclidean
        if metric=='jaccard':
            self.metric = jaccard
    
    def train_predict(self, X_train, y_train, include_self='False', distances=None):
        """
        RaRF regression on a training set

        Parameters
        ----------
        X_train: Array of training parameters
        y_train: Array of training responses
        include_self: For a given reaction, include itself in the training set. If False,
        equivalent to LOO.
        distances: Precomputed distances between training set reactions
        """
        
        predictions = []
        neighbours = []

        print("Training model...")
        for rxn_index,rxn in enumerate(tqdm(X_train)):

            indexes = []

            if distances is not None:

                if include_self:
                    indexes = np.where(distances[rxn_index, :] <= self.radius)[0]
                else:
                    indexes = np.where(distances[rxn_index, :] <= self.radius)[0]
                    indexes = indexes[indexes != rxn_index]
            
            else:
                if include_self:
                    for index,train_rxn in enumerate(X_train):
                        distance = self.metric(rxn,train_rxn)
                        if distance <= self.radius:
                                indexes.append(index)

                else:
                    for index,train_rxn in enumerate(X_train):
                        distance = self.metric(rxn,train_rxn)
                        if distance <= self.radius:
                            if rxn_index != index:
                                indexes.append(index)
            
            X_train_red = X_train[indexes]
            y_train_red = y_train[indexes]

            if len(X_train_red) == 0:
                print("NO NEIGHBOURS DETECTED, PREDICTING nan")
                predictions.append(np.nan)
            elif X_train_red.ndim == 1:
                predictions.append(y_train_red)
            else:
                model = RandomForestRegressor().fit(X_train_red,y_train_red)
                predictions.append(float(model.predict(rxn.reshape(1,-1))))
                
            neighbours.append(len(y_train_red))
        return predictions, neighbours
        
    def predict(self, X_train, y_train, X_test, distances): 
        """
        Train on known values and predict unknown values. Currently only configured for n_neighbours=int.
        
        Parameters
        ----------
        X_train: Array of training parameters
        y_train: Array of training responses
        X_test: Arry of test parameters
        distances: Precomputed distances between test and training set reactions
        """
        predictions = []
        neighbours = []

        print("Predicting values...")
        for test_index, test_rxn in enumerate(tqdm(X_test)):

            indexes = np.where(distances[test_index, :] <= self.radius)[0]
            X_train_red = X_train[indexes]
            y_train_red = y_train[indexes]

            if len(X_train_red) == 0:
                print("NO NEIGHBOURS DETECTED, PREDICTING nan")
                predictions.append(np.nan)
            elif X_train_red.ndim == 1:
                predictions.append(y_train_red)
            else:
                model = RandomForestRegressor().fit(X_train_red,y_train_red)
                predictions.append(float(model.predict(test_rxn.reshape(1,-1))))

            neighbours.append(len(y_train_red))

        return np.array(predictions), neighbours
    
    # what actually goes on here? Oh, X_train already removes stuff. Yay!
    def predict_parallel(self,X_train, y_train, X_test, distances, n_jobs=-1):
        """ 
        Parallel version of predict function

        Parameters
        ----------
        X_train: Array of training parameters
        y_train: Array of training responses
        X_test: Array of test parameters
        distances: Precomputed distances between training set reactions
        n_jobs: Number of parallel jobs to run
        """

        def process(test_index, test_rxn):
            indexes = np.where(distances[test_index, :] <= self.radius)[0]
            X_train_red = X_train[indexes]
            y_train_red = y_train[indexes]

            if len(X_train_red) == 0:
                return np.nan, len(y_train_red), np.array([])
            
            elif X_train_red.ndim == 1:
                return y_train_red, len(y_train_red), y_train_red
            else:
                model = RandomForestRegressor().fit(X_train_red, y_train_red)
                return float(model.predict(test_rxn.reshape(1, -1))), len(y_train_red), y_train_red

        results = Parallel(n_jobs=n_jobs)(delayed(process)(test_index, test_rxn) for test_index, test_rxn in enumerate(X_test))




        predictions, neighbours, neighbour_list = zip(*results)
        predictions = np.array(predictions)
        neighbours = np.array(neighbours)


        return predictions, neighbours, neighbour_list
    
    def train_parallel(self, X_train, y_train, include_self='False', distances=None, n_jobs=-1):
        """
        Parallel version of train_predict function

        Parameters
        ----------
        X_train: Array of training parameters
        y_train: Array of training responses
        include_self: For a given reaction, include itself in the training set. If False,
        equivalent to LOO.
        distances: Precomputed distances between training set reactions
        n_jobs: Number of parallel jobs to run
        """
        def process(rxn_index, rxn):
            indexes = []

            if distances is not None:

                if include_self:
                    indexes = np.where(distances[rxn_index, :] <= self.radius)[0]
                else:
                    indexes = np.where(distances[rxn_index, :] <= self.radius)[0]
                    indexes = indexes[indexes != rxn_index]
            
            else:
                if include_self:
                    for index,train_rxn in enumerate(X_train):
                        distance = self.metric(rxn,train_rxn)
                        if distance <= self.radius:
                                indexes.append(index)

                else:
                    for index,train_rxn in enumerate(X_train):
                        distance = self.metric(rxn,train_rxn)
                        if distance <= self.radius:
                            if rxn_index != index:
                                indexes.append(index)
            
            X_train_red = X_train[indexes]
            y_train_red = y_train[indexes]

            if len(X_train_red) == 0:
                return np.nan, len(y_train_red)
            elif X_train_red.ndim == 1:
                return y_train_red, len(y_train_red)
            else:
                model = RandomForestRegressor().fit(X_train_red,y_train_red)
                return float(model.predict(rxn.reshape(1,-1))), len(y_train_red)
        
        results = Parallel(n_jobs=n_jobs)(delayed(process)(rxn_index, rxn) for rxn_index, rxn in enumerate(X_train))

        predictions, neighbours = zip(*results)
        predictions = np.array(predictions)
        neighbours = np.array(neighbours)

        return predictions, neighbours
    
    def run_and_plot(i, X_train, y_train, X_test, y_test, distances):
        radius_pred, train_neighbours = RaRFRegressor.RaRFRegressor(radius=i, metric='jaccard').train_parallel(X_train,y_train, include_self='True')
        radius_testpred, test_neighbours, test_neighbours_list = RaRFRegressor.RaRFRegressor(radius=i,metric='jaccard').predict_parallel(X_train, y_train, X_test, distances)    

        test_neighbours = np.array(test_neighbours)
        nan_indexes = []
        index = -1
        for prediction in radius_testpred:
            index +=1
            if np.isnan(prediction) == True:
                nan_indexes.append(index)
            
        radius_testpred_temp = np.delete(radius_testpred,nan_indexes)
        y_test_temp = np.delete(y_test,nan_indexes)


        fig, (ax1, ax2) = plt.subplots(1,2, figsize=(10,5), dpi=300)


        ax1.plot(y_train,y_train, color='grey', zorder=0)
        ax1.scatter(y_train,radius_pred, label='train R2 ' + str(round(r2_score(y_train,radius_pred),2)), color='#279383')
        ax1.scatter(y_test_temp,radius_testpred_temp, label='test R2 ' + str(round(r2_score(y_test_temp,radius_testpred_temp,),2)), color='white', edgecolor='#279383')

        ax1.set_xlabel('Measured $\Delta\Delta G^‡$ (kcal/mol)')
        ax1.set_ylabel('Predicted $\Delta\Delta G^‡$ (kcal/mol)')
        ax1.legend()


        ax2 = sns.kdeplot(data=[[train_neighbours[x] for x in np.nonzero(train_neighbours)[0]], [test_neighbours[x] for x in np.nonzero(test_neighbours)[0]]], palette=[COLORA, COLORB])
        ax2.legend(['train', 'test'])
        ax2.set_xlim(-10,200)
        ax2.set_xlabel('# of neighbours')

        fig.suptitle(f'Radius {i}, {len(nan_indexes)}/{len(radius_testpred)} NaNs')
        plt.tight_layout()
        plt.show()

        return (mean_absolute_error(y_test_temp,radius_testpred_temp)), len(nan_indexes), np.average([test_neighbours[x] for x in np.nonzero(test_neighbours)[0]])

