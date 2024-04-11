import numpy as np

class Node():
    def __init__(self, data, feature_idx, feature_val, prediction_probs, information_gain) -> None:
        self.data = data
        self.feature_idx = feature_idx
        self.feature_val = feature_val
        self.prediction_probs = prediction_probs
        self.information_gain = information_gain
        self.feature_importance = self.data.shape[0] * self.information_gain
        self.left = None
        self.right = None

    def node_def(self) -> str:

        if (self.left or self.right):
            return f"❓ \033[1mNODE\033[0m | IG = {self.information_gain} | Split IF X[{self.feature_idx}] < {self.feature_val} then LEFT else RIGHT"
        else:
            unique_values, value_counts = np.unique(self.data[:,-1], return_counts=True)
            output = ", ".join([f"{value}->{count}" for value, count in zip(unique_values, value_counts)])            
            return f"🍃 \033[1mLEAF\033[0m | Label Counts = {output} | Pred Probs = {self.prediction_probs}"