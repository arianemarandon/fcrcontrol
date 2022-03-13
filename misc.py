#####################################################################################################
# False Clustering Rate computation
#####################################################################################################

import numpy as np
from itertools import permutations

def get_fcp(selection, labels_true, labels_preds):
    n  = len(labels_true)
    n_classes = len(np.unique(labels_true))
    n_sel = len(selection)

    #compute permutation that maximizes the accuracy   
    permumax = get_permutation(labels_true, labels_preds)
    
    labels_preds = np.eye(n_classes)[labels_preds][:,list(permumax)] #update labels_preds to match labels_true as much as possible 
    labels_preds  = np.argmax(labels_preds, 1)
    
    acc = np.sum(labels_true == labels_preds)/n 
    
    false_preds = np.nonzero(labels_true != labels_preds)[0]
    sel_false = len(set(selection) & set(false_preds.tolist()))

    fcp = sel_false / n_sel if n_sel > 0 else 0 
    fsel = n_sel/n 

    return fcp, acc, fsel


def get_permutation(y1, y2):
    n_classes = np.maximum(len(np.unique(y1)), len(np.unique(y2)))

    matchmax = np.NINF
    permumax = None
    for p in permutations(list(range(n_classes))):
        try: 
            y2_one_hot = np.eye(n_classes)[y2]
        except IndexError:
            print(n_classes)
            exit()
        y2_one_hot_permuted = y2_one_hot[:, list(p)]
        y2_permuted = np.argmax(y2_one_hot_permuted, 1)
        match = np.sum(y1 == y2_permuted)
        if match > matchmax: 
            permumax = p
            matchmax = match
    return permumax
