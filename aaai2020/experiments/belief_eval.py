
import math

from enum import Enum

import matplotlib.pyplot as plt

import itertools
import numpy as np
from scipy.special import logsumexp as lse

from itertools import combinations, chain
from scipy.special import comb

from scipy.spatial import ConvexHull, Delaunay

from belief_utils import comb_index, entropy, marginal_entropy

from belief import process_ctx, Belief, OrBelief
from cog_belief import CostBelief, EgoCostBelief, PriorType

np.seterr(all="raise")

def rollout(ctx, ids, belief, response_strategy):
    print(f"Running {type(belief)} with {response_strategy}")

    belief_type = None
    if isinstance(belief, EgoCostBelief):
        belief_type = f"EgoCostBelief"
    elif isinstance(belief, CostBelief):
        belief_type = f"CostBelief_priortype{belief.prior_type}"
    elif isinstance(belief, OrBelief):
        belief_type = f"OrBelief"
    else:
        raise ValueError

    sc = belief.sc
    xy = belief.xy

    N = 5
    fig, ax = plt.subplots(1, N, figsize=(4*N, 4))

    prior = belief.prior
    for n in range(N):
        EdHs = belief.compute_EdHs(prior)
        #mEdHs = belief.compute_marginal_EdHs(prior).max(-1)
        utt = belief.configs[EdHs.argmax()]
        #print("utt", belief.configs[EdHs.argmax()])
        #print("marg utt", belief.configs[mEdHs.argmax()])

        uttb = utt.astype(bool)
        ax[n].scatter(
            xy[:,0], xy[:,1],
            marker='o',
            s=50*(1+sc[:,0]),
            #c=-80*(sc[:,1]),
            #s = 50*(ctx[:,2] + ctx[:,2].min() + 1),
            c = -ctx[:,3],
            cmap="binary",
            edgecolor="black",
            linewidth=1,
        )
        ax[n].scatter(xy[uttb,0], xy[uttb,1], marker="x", s=100, c="r")
        for i, id in enumerate(ids):
            ax[n].annotate(id, (xy[i,0]+.025, xy[i,1]+.025))

        response = None
        if response_strategy == "all_yes":
            response = 1
        elif response_strategy == "all_no":
            response = 0
        elif response_strategy == "alternate":
            response = 1 if n % 2 == 0 else 0
        else:
            raise ValueError

        print("prior", belief.marginals(prior))
        print(response)
        new_prior = belief.posterior(prior, utt, response)
        print("posterior", belief.marginals(new_prior))
        #import pdb; pdb.set_trace()

        belief.history.append(utt)
        prior = new_prior

    plt.savefig(f"plan_plots/{belief_type}_{response_strategy}.png")


def main():
    num_dots = 7

    # scenario S_pGlR0nKz9pQ4ZWsw
    # streamlit run main.py
    ctx = np.array([
        0.635, -0.4,   2/3, -1/6,  # 8
        0.395, -0.7,   0.0,  3/4,  # 11
        -0.74,  0.09,  2/3, -2/3,  # 13
        -0.24, -0.63, -1/3, -1/6,  # 15
        0.15,  -0.58,  0.0,  0.24, # 40
        -0.295, 0.685, 0.0, -8/9,  # 50
        0.035, -0.79, -2/3,  0.56, # 77
    ], dtype=float).reshape(-1, 4)
    ids = np.array(['8', '11', '13', '15', '40', '50', '77'], dtype=int)

    # reflect across y axis
    ctx[:,1] = -ctx[:,1]

    beliefs = [
        CostBelief(num_dots, ctx, num_size_buckets=5, num_color_buckets=5),
        CostBelief(num_dots, ctx, num_size_buckets=5, num_color_buckets=5, prior_type=PriorType.ISING),
        CostBelief(num_dots, ctx, num_size_buckets=5, num_color_buckets=5, prior_type=PriorType.MST),
        EgoCostBelief(num_dots, ctx, num_size_buckets=5, num_color_buckets=5),
        OrBelief(num_dots, ctx, num_size_buckets=5, num_color_buckets=5),
    ]
    strategies = [
        "all_yes",
        "all_no",
        "alternate",
    ]
    for belief in beliefs:
        belief.ids = ids
        for strategy in strategies:
            rollout(ctx, ids, belief, strategy)


if __name__ == "__main__":
    main()
