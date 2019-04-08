from sklearn.preprocessing import MinMaxScaler
import numpy as np

def softmax(x):
    return MinMaxScaler().fit_transform(np.array(x).reshape(-1, 1)).flatten()

