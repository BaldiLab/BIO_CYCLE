from typing import List, Any, Tuple
import numpy as np
import math
import os
import pandas as pd


def flatten(data: List[List[Any]]) -> Tuple[List[Any], List[int]]:
    num_each = [len(x) for x in data]
    split_idxs: List[int] = list(np.cumsum(num_each)[:-1])

    data_flat = [item for sublist in data for item in sublist]

    return data_flat, split_idxs


def makedirs_if_needed(dname: str):
    if (len(dname) > 0) and not os.path.exists(dname):
        os.makedirs(dname)


def split_evenly(num_total: int, num_splits: int) -> List[int]:
    num_per: List[int] = [math.floor(num_total / num_splits) for _ in range(num_splits)]
    left_over: int = num_total % num_splits
    for idx in range(left_over):
        num_per[idx] += 1

    return num_per


def bhq(pvals: np.array):
    i = np.arange(1, len(pvals) + 1)[::-1]
    idxs_decr = np.argsort(pvals)[::-1]
    idxs_rev = np.argsort(idxs_decr)

    qvals = pvals[idxs_decr] * len(pvals)/i
    qvals = pd.Series(qvals).cummin().to_numpy()
    qvals = np.minimum(qvals, 1)

    qvals = qvals[idxs_rev]

    return qvals


def sin_cos_to_phase(sines, cosines):
    phases = np.zeros(sines.shape[0])

    sin_pos_idxs = np.where(sines >= 0)[0]
    sin_neg_idxs = np.where(sines < 0)[0]

    phases[sin_pos_idxs] = np.arccos(cosines[sin_pos_idxs])
    phases[sin_neg_idxs] = np.arccos(-cosines[sin_neg_idxs]) + np.pi

    return phases
