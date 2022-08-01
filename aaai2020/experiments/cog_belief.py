
import math

from enum import Enum

import itertools
import numpy as np
from scipy.special import logsumexp as lse

from itertools import combinations, chain
from scipy.special import comb

from scipy.spatial import ConvexHull, Delaunay

from belief_utils import comb_index, entropy, marginal_entropy

from belief import process_ctx, Belief, OrBelief


class CostBelief(OrBelief):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_denies = np.array([
            self.spatial_deny(x, self.ctx) for x in self.configs
        ])

    def spatial_deny(self, x, ctx):
        #return .001
        if x.sum() == 0:
            return 0.001

        rg = np.arange(self.num_dots)
        xy = self.xy
        pairs = np.array(list(itertools.product(rg, rg)))
        xy_pairs = xy[pairs].reshape((self.num_dots, self.num_dots, 2, 2))
        dist_pairs = np.linalg.norm(xy_pairs[:,:,0] - xy_pairs[:,:,1], axis=-1)
        idxs = dist_pairs.argsort()
        ranks = idxs.argsort()+1
        #ranks = np.empty_like(idxs)
        #ranks[idxs] = np.arange(self.num_dots)

        dots = x.nonzero()[0]

        dot_pair_ranks = ranks[dots[:,None], dots]
        scores = dot_pair_ranks.sum(-1)
        score = scores.min()

        denominator = np.arange(self.num_dots - len(dots)+1, self.num_dots+1).sum()
        #print(dots)
        #print(score)
        #print(denominator)
        #print(score / denominator)
        #return score / denominator
        return np.exp(score - denominator)

    def temporal_deny(self, x, history):
        return .001
        max_step = 3
        weight = 1
        max_dots = 2

        denominator = max_dots * weight * max_step
        if len(self.history) == 0:
            numerator = x.sum() * weight * max_step
            return (np.exp(numerator - denominator)).clip(0, 1)

        history = np.stack(self.history)[::-1]
        past = history.sum(0)
        dots = x.nonzero()[0]
        is_new = past[dots] == 0
        distance = 1 + history[:,x.astype(bool)].argmax(0)

        numerator = (
            is_new * weight * max_step
            + np.minimum(distance, max_step) * weight * ~is_new
        ).sum()

        return (np.exp(numerator - denominator)).clip(0, 1)

    def joint(self, prior, utt):
        # p(r | u,s)
        # prior: num_configs * 7
        # p(r=0|u,s)p(s)
        # = \sum_z p(s)p(z|s) p(r=0|u,s)p(r=0|u,z)
        # = p(s)p(r=0|u,s)\sum_z p(z|s)p(r=0|u,z)
        # = p(s)(1 - p(r=1|u,s)) |z|C|u|9^-|u|

        temporal_denies = np.array([
            self.temporal_deny(x, self.history) for x in self.configs
        ])
        p_r1 = []
        p_r0 = []
        for s,ps in enumerate(prior):
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            u = int(utt.sum())
            utt_idx = np.right_shift(np.packbits(utt), 8-self.num_dots)
            c_likelihood = self.config_likelihood[utt_idx,s].item()
            td = temporal_denies[s]
            sd = self.spatial_denies[s]
            likelihood = c_likelihood * (1-td) * (1-sd)
            #import pdb; pdb.set_trace()
            #for i,d in enumerate(utt):
                #if d == 1:
                    #likelihood *= self.likelihood[1,d,state_config[i]]
            distractor_prob = 1 - comb(z,u) * 9. ** (-u)

            #p_deny = (1 - likelihood)*distractor_prob
            p_deny = (1 - likelihood)*distractor_prob

            #print(p_deny)
            if p_deny > 1:
                print(likelihood)
                print(distractor_prob)
                print(td)
                print(sd)
                import pdb; pdb.set_trace()

            p_r0.append(p_deny * ps)
            p_r1.append((1 - p_deny) * ps)
        return np.array((p_r0, p_r1))


if __name__ == "__main__":
    np.seterr(all="raise")

    num_dots = 7

    print("spatial context tests")
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

    belief = CostBelief(num_dots, ctx, num_size_buckets=5, num_color_buckets=5)
    sc = belief.sc
    xy = belief.xy
    #utt_feats = belief.get_feats(utt)
    #matches = belief.resolve_utt(*utt_feats)

    utt = np.array([1,0,0,1,0,1,1])
    uttb = utt.astype(bool)

    print("distance tests")
    diameters = belief.compute_diameters()
    contiguity = belief.compute_contiguity()

    # marginal entropy tests
    marginal = belief.marginals(belief.prior)
    marginals = np.stack((1-marginal, marginal), -1)

    new_weight = 5
    distance_weight = 1

    N = 5
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(N)

    prior = belief.prior
    for n in range(N):
        EdHs = belief.compute_EdHs(prior)
        mEdHs = belief.compute_marginal_EdHs(prior).max(-1)
        # looks like the marginal posterior wants to improve the probability
        # of one dot by asking about ALL OTHERS
        mps = belief.compute_marginal_posteriors(prior)

        costs = belief.compute_processing_costs(new_weight, distance_weight)
        #costs = costs / 25 / 7
        costs = costs / belief.num_dots / 2
        #utt = belief.configs[(mEdHs + costs).argmax()]
        #utt = belief.configs[(EdHs + costs).argmax()]
        utt = belief.configs[EdHs.argmax()]
        #print(belief.configs[(mEdHs + costs).argmax()])
        #print(belief.configs[(EdHs + costs).argmax()])
        print(utt)
        #print(belief.configs[mEdHs.argmax()])

        print("prior", belief.marginals(prior))
        #new_prior = belief.posterior(prior, utt, 1 if n % 2 == 0 else 0)
        new_prior = belief.posterior(prior, utt, 1)
        print("posterior", belief.marginals(new_prior))
        #import pdb; pdb.set_trace()

        belief.history.append(utt)
        prior = new_prior

        #import pdb; pdb.set_trace()
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
    plt.show()

    utt = np.array([1,0,0,1,0,1,1])
    uttb = utt.astype(bool)
    a = belief.spatial_deny(utt, ctx)
    b = belief.temporal_deny(utt, belief.history)

    import pdb; pdb.set_trace()

