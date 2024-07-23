import numpy as np
import pandas as pd 
from sklearn.tree import DecisionTreeRegressor 
from sklearn.preprocessing import OneHotEncoder

class GradientBoostingClassifierFromScratch():
    '''Gradient Boosting Classifier from Scratch.
    
    Parameters
    ----------
    n_estimators : int
        number of boosting rounds
        
    learning_rate : float
        learning rate hyperparameter
        
    max_depth : int
        maximum tree depth
    '''
    
    def __init__(self, n_estimators, learning_rate=0.1, max_depth=1):
        self.n_estimators=n_estimators; 
        self.learning_rate=learning_rate
        self.max_depth=max_depth;
    
    def fit(self, X, y):
        '''Fit the GBM
        
        Parameters
        ----------
        X : ndarray of size (number observations, number features)
            design matrix
            
        y : ndarray of size (number observations,)
            integer-encoded target labels in {0,1,...,k-1}
        '''
        
        self.n_classes = pd.Series(y).nunique()
        y_ohe = self._one_hot_encode_labels(y)

        raw_predictions = np.zeros(shape=y_ohe.shape)
        probabilities = self._softmax(raw_predictions)
        self.boosters = []
        for m in range(self.n_estimators):
            class_trees = []
            for k in range(self.n_classes):
                negative_gradients = self._negative_gradients(y_ohe[:, k], probabilities[:, k])
                hessians = self._hessians(probabilities[:, k])
                tree = DecisionTreeRegressor(max_depth=self.max_depth)
                tree.fit(X, negative_gradients);
                self._update_terminal_nodes(tree, X, negative_gradients, hessians)
                raw_predictions[:, k] += self.learning_rate * tree.predict(X)
                probabilities = self._softmax(raw_predictions)
                class_trees.append(tree)
            self.boosters.append(class_trees)
    
    def _one_hot_encode_labels(self, y):
        if isinstance(y, pd.Series): y = y.values
        ohe = OneHotEncoder()
        y_ohe = ohe.fit_transform(y.reshape(-1, 1)).toarray()
        return y_ohe
        
    def _negative_gradients(self, y_ohe, probabilities):
        return y_ohe - probabilities
    
    def _hessians(self, probabilities): 
        return probabilities * (1 - probabilities)

    def _softmax(self, raw_predictions):
        numerator = np.exp(raw_predictions) 
        denominator = np.sum(np.exp(raw_predictions), axis=1).reshape(-1, 1)
        return numerator / denominator
        
    def _update_terminal_nodes(self, tree, X, negative_gradients, hessians):
        '''Update the terminal node predicted values'''
        # terminal node id's
        leaf_nodes = np.nonzero(tree.tree_.children_left == -1)[0]
        # compute leaf for each sample in ``X``.
        leaf_node_for_each_sample = tree.apply(X)
        for leaf in leaf_nodes:
            samples_in_this_leaf = np.where(leaf_node_for_each_sample == leaf)[0]
            negative_gradients_in_leaf = negative_gradients.take(samples_in_this_leaf, axis=0)
            hessians_in_leaf = hessians.take(samples_in_this_leaf, axis=0)
            val = np.sum(negative_gradients_in_leaf) / np.sum(hessians_in_leaf)
            tree.tree_.value[leaf, 0, 0] = val
          
    def predict_proba(self, X):
        '''Generate probability predictions for the given input data.'''
        raw_predictions =  np.zeros(shape=(X.shape[0], self.n_classes))
        for k in range(self.n_classes):
            for booster in self.boosters:
                raw_predictions[:, k] +=self.learning_rate * booster[k].predict(X)
        probabilities = self._softmax(raw_predictions)
        return probabilities
        
    def predict(self, X):
        '''Generate predicted labels (as 1-d array)'''
        probabilities = self.predict_proba(X)
        return np.argmax(probabilities, axis=1)
    





