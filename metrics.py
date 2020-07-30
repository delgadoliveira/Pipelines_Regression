import numpy as np
def get_PE(pred, sold):
    return np.divide(np.subtract(pred, sold),sold)

def get_MdPE(pred, sold):
    return np.median(get_PE(pred, sold))

def get_MdAPE(pred, sold):
    return np.median(np.abs(get_PE(pred, sold)))

def get_PPE(pred, sold, ppe):
    APE = np.abs(get_PE(pred, sold))
    return np.divide(np.count_nonzero(APE <= ppe), len(APE))

def scorer_get_PPE(sold, pred):
    APE = np.abs(get_PE(pred, sold))
    return np.divide(np.count_nonzero(APE <= 0.2), len(APE))
