
import math

from enum import Enum

import itertools
import numpy as np
from scipy.special import logsumexp as lse

from itertools import combinations, chain
from scipy.special import comb

from scipy.spatial import ConvexHull, Delaunay

def comb_index(n, k):
    count = comb(n, k, exact=True)
    index = np.fromiter(
        chain.from_iterable(combinations(range(n), k)), 
        int,
        count=count*k,
    )
    return index.reshape(-1, k)

def safe_log(x, eps=1e-10):
    result = np.where(x > eps, x, 0)
    np.log(result, out=result, where=result > 0)
    return result

# discrete entropy
def entropy(px):
    Hx = px * safe_log(px)
    return -(Hx).sum(-1)

# entropy for computing dot marginals
# px: num_dots
def marginal_entropy(px):
    px = np.stack((1-px, px), -1)
    return entropy(px)
