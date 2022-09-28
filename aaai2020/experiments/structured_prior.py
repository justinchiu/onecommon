import matplotlib.pyplot as plt

import numpy as np
from scipy.special import logsumexp as lse

# ising model prior based on distance

num_dots = 4

configs = np.array([
    np.unpackbits(np.array([x], dtype=np.ubyte))[8-num_dots:]
    for x in range(2 ** num_dots)
])
rad_configs = np.where(configs == 0, -1, configs)

xy = np.random.rand(num_dots,2) * 2 - 1
dists = ((xy[:,None] - xy[None]) ** 2).sum(-1)

def ising_prior(configs, dists, tau=3):
    log_unnormalized_prior = np.einsum("di,ij,dj->d", rad_configs, -dists, rad_configs) / tau
    Z = lse(log_unnormalized_prior)
    log_prior = log_unnormalized_prior - Z
    prior = np.exp(log_prior)
    return prior

def visualize(configs, prior, name=None):
    fig, ax = plt.subplots(1, 2**num_dots, figsize=(3*(2**num_dots), 2))
    for i, (config, prob) in enumerate(zip(configs, prior)):
        bconfig = config.astype(bool)
        ax[i].scatter(
            xy[bconfig,0], xy[bconfig,1], marker="o", s=100,
        )
        ax[i].set_title(f"Prob {prob:.2f}")
        ax[i].set_xlim(-1, 1)
        ax[i].set_ylim(-1, 1)
    if name is not None:
        plt.savefig(name)
    else:
        plt.show()
    plt.clf()

#visualize(configs, ising_prior(configs, dists))

def mst_rec(num_dots, dist_pairs, dots, remaining_dots, score, edges):
    if len(remaining_dots) == 0:
        return score, edges
    trunc_dists = dist_pairs[
        dots[:,None],
        remaining_dots,
    ]
    src, tgt = np.unravel_index(np.argmin(trunc_dists), trunc_dists.shape)
    distance = trunc_dists[src, tgt]

    best_dot = remaining_dots[tgt]
    idx = np.where(remaining_dots == best_dot)[0].item()
    return mst_rec(
        num_dots,
        dist_pairs,
        np.append(dots, best_dot),
        np.delete(remaining_dots, idx),
        score + distance,
        edges + [(dots[src], best_dot)],
    )

score, edges = mst_rec(
    num_dots, dists,
    np.array([0]),
    np.array([1,2,3]),
    0,
    [],
)
def mst_prior(configs, dists):
    log_unnormalized_prior = np.array([
        -mst_rec(num_dots, dists, x.nonzero()[0][:1], x.nonzero()[0][1:], 0, [])[0] / 2
        for x in configs
    ])
    Z = lse(log_unnormalized_prior)
    log_prior = log_unnormalized_prior - Z
    prior = np.exp(log_prior)
    return prior

visualize(configs, mst_prior(configs, dists))


