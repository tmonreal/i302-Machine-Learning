import numpy as np
import matplotlib.pyplot as plt
from collections import Counter
from treenode import Node

class DecisionTree():
    def __init__(self, 
                 max_depth=4, 
                 min_samples_leaf=1, 
                 min_information_gain=0.0, 
                 numb_of_features_splitting=None) -> None:
        """
        Constructor function for DecisionTree instance
        Inputs:
            max_depth (int): max depth of the tree
            min_samples_leaf (int): min number of samples required to be in a leaf 
                                    to make the splitting possible
            min_information_gain (float): min information gain required to make the 
                                          splitting possible
            num_of_features_splitting (str):  number of features (n_features) to consider 
                                              when looking for the best split:
                                            - if sqrt then num_of_features_splitting = sqrt(n_features), 
                                            - if log then num_of_features_splitting = log2(n_features),
                                            - if None then num_of_features_splitting = n_features.                                     
        """
        self.max_depth = max_depth
        self.min_samples_leaf = min_samples_leaf
        self.min_information_gain = min_information_gain
        self.numb_of_features_splitting = numb_of_features_splitting

    def entropy(self, class_probabilities: list) -> float:
        return sum([-p * np.log2(p) for p in class_probabilities if p>0])
    
    def class_probabilities(self, labels: list) -> list:
        total_count = len(labels)
        return [label_count / total_count for label_count in Counter(labels).values()]

    def data_entropy(self, labels: list) -> float:
        return self.entropy(self.class_probabilities(labels))
    
    def partition_entropy(self, subsets: list) -> float:
        """
        Calculates the entropy of a partitioned dataset. 
        Inputs:
            - subsets (list): list of label lists 
            (Example: [[1,0,0], [1,1,1] represents two subsets 
            with labels [1,0,0] and [1,1,1] respectively.)

        Returns:
            - Entropy of the labels
        """
        # Total count of all labels across all subsets.
        total_count = sum([len(subset) for subset in subsets]) 
        # Calculates entropy of each subset and weights it by its proportion in the total dataset 
        return sum([self.data_entropy(subset) * (len(subset) / total_count) for subset in subsets])
    
    def split(self, data: np.array, feature_idx: int, feature_val: float) -> tuple:
        """
        Partitions the dataset into two groups based on a specified feature 
        and its corresponding threshold value.
        Inputs:
        - data (np.array): training dataset
        - feature_idx (int): feature used to split
        - feature_val (float): threshold value 
        """
        mask_below_threshold = data[:, feature_idx] < feature_val
        group1 = data[mask_below_threshold]
        group2 = data[~mask_below_threshold]

        return group1, group2
    
    def select_features_to_use(self, data: np.array) -> list:
        """
        Randomly selects the subset of features to use while 
        splitting with respect to hyperparameter numb_of_features_splitting

        Inputs:
        - data (np.array): numpy array with training data.
        Returns:
        - feature_idx_to_use(np.array): numpy array with feature 
        indexes to be used during split.
        """
        feature_idx = list(range(data.shape[1]-1))

        if self.numb_of_features_splitting == "sqrt":
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.sqrt(len(feature_idx))))
        elif self.numb_of_features_splitting == "log":
            feature_idx_to_use = np.random.choice(feature_idx, size=int(np.log2(len(feature_idx))))
        else:
            feature_idx_to_use = feature_idx

        return feature_idx_to_use
        
    def find_best_split(self, data: np.array) -> tuple:
        """
        Finds the optimal feature and value to split the dataset on 
        at each node of the tree (with the lowest entropy).
        Inputs:
            - data (np.array): numpy array with training data
        Returns:
            - 2 splitted groups (g1_min, g2_min) and split information 
            (min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy)
        """
        min_part_entropy = 1e9
        feature_idx_to_use =  self.select_features_to_use(data)

        for idx in feature_idx_to_use: # For each feature to try
            feature_vals = np.percentile(data[:, idx], q=np.arange(25, 100, 25)) # Calc 25th, 50th, and 75th percentiles
            for feature_val in feature_vals: # For each percentile value we partition in 2 groups
                g1, g2, = self.split(data, idx, feature_val)
                part_entropy = self.partition_entropy([g1[:, -1], g2[:, -1]]) # Calculate entropy of that partition
                if part_entropy < min_part_entropy:
                    min_part_entropy = part_entropy
                    min_entropy_feature_idx = idx
                    min_entropy_feature_val = feature_val
                    g1_min, g2_min = g1, g2

        return g1_min, g2_min, min_entropy_feature_idx, min_entropy_feature_val, min_part_entropy

    def find_label_probs(self, data: np.array) -> np.array:
        """
        Computes the distribution of labels in the dataset.
        It returns the array label_probabilities, which contains 
        the probabilities of each label occurring in the dataset.

        Inputs:
            - data (np.array): numpy array with training data
        Returns:
            - label_probabilities (np.array): numpy array with the
            probabilities of each label in the dataset.
        """
        # Transform labels to ints (assume label in last column of data)
        labels_as_integers = data[:,-1].astype(int)
        # Calculate the total number of labels
        total_labels = len(labels_as_integers)
        # Calculate the ratios (probabilities) for each label
        label_probabilities = np.zeros(len(self.labels_in_train), dtype=float)
        # Populate the label_probabilities array based on the specific labels
        for i, label in enumerate(self.labels_in_train):
            label_index = np.where(labels_as_integers == i)[0]
            if len(label_index) > 0:
                label_probabilities[i] = len(label_index) / total_labels

        return label_probabilities

    def create_tree(self, data: np.array, current_depth: int) -> Node:
        """
        Recursive, depth first tree creation algorithm.
        Inputs:
            - data (np.array): numpy array with training data
            - current_depth (int): current depth of the recursive tree
        Returns:
            - node (Node): current node, which contains references to its left and right child nodes.
        """
        # Check if the max depth has been reached (stopping criteria)
        if current_depth > self.max_depth:
            return None
        # Find best split
        split_1_data, split_2_data, split_feature_idx, split_feature_val, split_entropy = self.find_best_split(data)
        # Find label probs for the node
        label_probabilities = self.find_label_probs(data)
        # Calculate information gain
        node_entropy = self.entropy(label_probabilities)
        information_gain = node_entropy - split_entropy
        # Create node
        node = Node(data, split_feature_idx, split_feature_val, label_probabilities, information_gain)
        # Check if the min_samples_leaf has been satisfied (stopping criteria)
        if self.min_samples_leaf > split_1_data.shape[0] or self.min_samples_leaf > split_2_data.shape[0]:
            return node
        # Check if the min_information_gain has been satisfied (stopping criteria)
        elif information_gain < self.min_information_gain:
            return node
        
        current_depth += 1
        node.left = self.create_tree(split_1_data, current_depth)
        node.right = self.create_tree(split_2_data, current_depth)
        
        return node
    
    def predict_one_sample(self, X: np.array) -> np.array:
        """
        Returns prediction for 1 dim array.
        """
        node = self.tree
        # Finds the leaf which X belongs to
        while node:
            pred_probs = node.prediction_probs
            if X[node.feature_idx] < node.feature_val:
                node = node.left
            else:
                node = node.right

        return pred_probs

    def train(self, X_train: np.array, Y_train: np.array) -> None:
        """
        Trains the model with given X and Y datasets.
        Inputs:
            - X_train (np.array): training features
            - Y_train (np.array): training labels
        """

        # Concat features and labels
        self.labels_in_train = np.unique(Y_train)
        train_data = np.concatenate((X_train, np.reshape(Y_train, (-1, 1))), axis=1)
        # Create tree
        self.tree = self.create_tree(data=train_data, current_depth=0)
        # Calculate feature importance
        self.feature_importances = dict.fromkeys(range(X_train.shape[1]), 0)
        self.calculate_feature_importance(self.tree)
        # Normalize the feature importance values
        self.feature_importances = {k: v / total for total in (sum(self.feature_importances.values()),) for k, v in self.feature_importances.items()}

    def predict_proba(self, X_set: np.array) -> np.array:
        """
        Returns the predicted probs for a given data set
        """

        pred_probs = np.apply_along_axis(self.predict_one_sample, 1, X_set)
        
        return pred_probs

    def predict(self, X_set: np.array) -> np.array:
        """
        Returns the predicted labels for a given data set
        """

        pred_probs = self.predict_proba(X_set)
        preds = np.argmax(pred_probs, axis=1)
        
        return preds    
        
    def print_recursive(self, node: Node, level=0) -> None:
        if node != None:
            self.print_recursive(node.left, level + 1)
            print('    ' * 4 * level + '→ ' + node.node_def())
            self.print_recursive(node.right, level + 1)

    def print_tree(self) -> None:
        self.print_recursive(node=self.tree)

    def calculate_feature_importance(self, node):
        """
        Calculates the feature importance by visiting each node in the tree recursively
        """
        if node != None:
            self.feature_importances[node.feature_idx] += node.feature_importance
            self.calculate_feature_importance(node.left)
            self.calculate_feature_importance(node.right)         

    def print_feature_importance(self):
        """
        Prints the feature importance values.
        """
        print("Feature Importance:")
        for feature_idx, importance in self.feature_importances.items():
            print(f"Feature {feature_idx}: Importance = {importance}")


    def plot_feature_importance(self, feature_names):
        """
        Plots the feature importance values with feature names.
        Inputs:
            - feature_names (list): List of feature names.
        """
        importance_values = list(self.feature_importances.values())

        plt.figure(figsize=(10, 6))
        plt.bar(feature_names, importance_values, color='blue')
        plt.xlabel('Feature Name')
        plt.ylabel('Importance')
        plt.title('Feature Importance')
        plt.xticks()
        plt.show()
