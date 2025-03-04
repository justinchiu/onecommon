
import math

from enum import Enum

import itertools
import numpy as np
from scipy.special import logsumexp as lse

from itertools import combinations, chain
from scipy.special import comb

from scipy.spatial import ConvexHull, Delaunay

from belief_utils import comb_index, entropy, marginal_entropy

from belief import process_ctx, Belief, OrBelief, PriorType


class CostBelief(OrBelief):
    """
    Or model for response modeling.
    Partner will (noisily) confirm an utterance if they see all dots mentioned
    OR have matching dots in unobserved context.
    The OR happens at the config level.
    * response r: 1
    * utterance u: num_dots
    * state s: num_dots
    * unobserved partner dots z: num_dots - |s|

    Normal model for dots and state
        p(r=1|u,s) = initialization
        p(r=0|u,s) = 1-p(r=1|u,s)
    Noisy-OR
        p(r=0|u,s,z) = 1-p(r=0|u,s)p(r=0|u,z)
    Dot distractors
        p(r=0|u,z) = 1 - |z|C|u| 9^-|u|

    Accurately estimates failure of small and large configurations.

    Note on p(r=0|u,z) = 1-|z|C|u|9^-|u|:
        color = light, medium, dark
        size = small, medium, dark
        Assume descriptions are all independent, so only 9 possibilities
        for each dot in z
        Size of z: remaining dots outside of s |z| = num_dots - |s|
    """
    def __init__(
        self,
        num_dots,
        ctx,
        correct = 0.95,
        overlap_size = None,
        absolute = True,
        use_diameter = False,
        use_contiguity = False,
        num_size_buckets = 5,
        num_color_buckets = 5,
        prior_type = PriorType.MST,
    ):
        super().__init__(
            num_dots,
            ctx,
            overlap_size = overlap_size,
            absolute = absolute,
            correct = correct,
            use_diameter = use_diameter,
            use_contiguity = use_contiguity,
            num_size_buckets = num_size_buckets,
            num_color_buckets = num_color_buckets,
            prior_type = prior_type,
        )

        # redo the dot likelihood
        self.spatial_resolvable = np.zeros((self.num_configs,), dtype=bool)
        for u, utt in enumerate(self.configs):
            self.spatial_resolvable[u] = self.is_contiguous(utt)
            for s, config in enumerate(self.configs):
                self.config_likelihood[u,s] = (
                    correct
                    if self.resolvable[u,s] and self.spatial_resolvable[u]
                    else 1 - correct
                )

    def is_contiguous(self, x):
        if x.sum() <= 1:
            return True

        rg = np.arange(self.num_dots)
        xy = self.xy
        pairs = np.array(list(itertools.product(rg, rg)))
        xy_pairs = xy[pairs].reshape((self.num_dots, self.num_dots, 2, 2))
        dist_pairs = np.linalg.norm(xy_pairs[:,:,0] - xy_pairs[:,:,1], axis=-1)
        idxs = dist_pairs.argsort()
        ranks = idxs.argsort()

        dots = x.nonzero()[0]

        def score_rec(dots, remaining_dots, score):
            if len(remaining_dots) == 0:
                return score
            remainder = np.delete(rg, dots)
            trunc_dists = dist_pairs[
                np.array(dots)[:,None],
                remainder,
            ]
            trunc_idxs = trunc_dists.argsort()
            trunc_ranks = trunc_idxs.argsort()

            dot_dists = dist_pairs[
                np.array(dots)[:,None],
                remaining_dots,
            ]
            closest_dots = remaining_dots[dot_dists.argmin(-1)]
            #best_ranks = ranks[np.array(dots), closest_dots]
            col, row = np.where(remainder[:,None] == closest_dots)
            best_ranks = trunc_ranks[row, col]
            best_rank = best_ranks.min()

            best_dot = closest_dots[best_ranks.argmin()]
            idx = np.where(remaining_dots == best_dot)[0].item()
            return score_rec(dots + [best_dot], np.delete(remaining_dots, idx), score + best_rank)

        scores = []
        for i, dot in enumerate(dots):
            remaining_dots = np.delete(dots, i)
            score = score_rec([dot], remaining_dots, 0)
            scores.append(score)

        return min(scores) == 0


class CostBeliefDeprecated(OrBelief):
    def __init__(self, *args, use_spatial=True, use_temporal=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.spatial_denies = np.array([
            self.spatial_deny(x, self.ctx) for x in self.configs
        ])
        self.use_spatial = use_spatial
        self.use_temporal = use_temporal

    def spatial_deny_old(self, x, ctx):
        #return .001
        if x.sum() <= 1:
            return 0.001

        rg = np.arange(self.num_dots)
        xy = self.xy
        pairs = np.array(list(itertools.product(rg, rg)))
        xy_pairs = xy[pairs].reshape((self.num_dots, self.num_dots, 2, 2))
        dist_pairs = np.linalg.norm(xy_pairs[:,:,0] - xy_pairs[:,:,1], axis=-1)
        idxs = dist_pairs.argsort()
        ranks = idxs.argsort()
        #ranks = np.empty_like(idxs)
        #ranks[idxs] = np.arange(self.num_dots)

        dots = x.nonzero()[0]

        dot_pair_ranks = ranks[dots[:,None], dots]
        scores = dot_pair_ranks.sum(-1)
        score = scores.min()

        # rank will always have 1 b/c dot is closest to itself, then other stuff
        # so highest rank should be other stuff = far away + 1
        # so bottom of range should start at 7 if there is only one dot,
        # 6 if there are two, etc
        # top of range = 7, since 7 dots
        denominator = np.arange(self.num_dots - len(dots)+1, self.num_dots).sum()
        """
        print("dots", dots)
        print("dot_pair_ranks", dot_pair_ranks)
        print("score", score)
        print("den", denominator)
        print("s/d", score / denominator)
        print("exp(s - d)", np.exp(score - denominator))
        """
        return (score / denominator).clip(0, 1)
        # linear scores seems better than log linear
        #import pdb; pdb.set_trace()
        #return np.exp(score - denominator).clip(0, 1)
        
    def spatial_deny(self, x, ctx):
        #return .001
        if x.sum() <= 1:
            return 0.001

        rg = np.arange(self.num_dots)
        xy = self.xy
        pairs = np.array(list(itertools.product(rg, rg)))
        xy_pairs = xy[pairs].reshape((self.num_dots, self.num_dots, 2, 2))
        dist_pairs = np.linalg.norm(xy_pairs[:,:,0] - xy_pairs[:,:,1], axis=-1)
        idxs = dist_pairs.argsort()
        ranks = idxs.argsort()
        #ranks = np.empty_like(idxs)
        #ranks[idxs] = np.arange(self.num_dots)

        dots = x.nonzero()[0]

        dot_pair_ranks = ranks[dots[:,None], dots]
        scores = dot_pair_ranks.sum(-1)
        score = scores.min()

        # rank will always have 1 b/c dot is closest to itself, then other stuff
        # so highest rank should be other stuff = far away + 1
        # so bottom of range should start at 7 if there is only one dot,
        # 6 if there are two, etc
        # top of range = 7, since 7 dots
        denominator = np.arange(self.num_dots - len(dots)+1, self.num_dots).sum()
        """
        print("dots", dots)
        print("dot_pair_ranks", dot_pair_ranks)
        print("score", score)
        print("den", denominator)
        print("s/d", score / denominator)
        print("exp(s - d)", np.exp(score - denominator))
        """

        old_score = (score / denominator).clip(0, 1)
        # linear scores seems better than log linear
        #import pdb; pdb.set_trace()
        #return np.exp(score - denominator).clip(0, 1)

        def score_rec(dots, remaining_dots, score):
            if len(remaining_dots) == 0:
                return score
            remainder = np.delete(rg, dots)
            trunc_dists = dist_pairs[
                np.array(dots)[:,None],
                remainder,
            ]
            trunc_idxs = trunc_dists.argsort()
            trunc_ranks = trunc_idxs.argsort()

            dot_dists = dist_pairs[
                np.array(dots)[:,None],
                remaining_dots,
            ]
            closest_dots = remaining_dots[dot_dists.argmin(-1)]
            #best_ranks = ranks[np.array(dots), closest_dots]
            col, row = np.where(remainder[:,None] == closest_dots)
            best_ranks = trunc_ranks[row, col]
            best_rank = best_ranks.min()

            #if len(dots) + len(remaining_dots) > 3 and best_rank < len(dots):
                #import pdb; pdb.set_trace()
            best_dot = closest_dots[best_ranks.argmin()]
            idx = np.where(remaining_dots == best_dot)[0].item()
            return score_rec(dots + [best_dot], np.delete(remaining_dots, idx), score + best_rank)

        scores = []
        for i, dot in enumerate(dots):
            remaining_dots = np.delete(dots, i)
            score = score_rec([dot], remaining_dots, 0)
            scores.append(score)

        #import pdb; pdb.set_trace()
        #return (min(scores) / denominator).clip(0, 1)
        #return (min(scores) * 3 / denominator).clip(0, 1)
        return 0.99 if min(scores) > 0 else 0.01
        #return 1.0 if min(scores) > 0 else 0.0
        # ^ HARD CONTIGUITY PENALTY



    def temporal_confirm(self, x, history):
        tau = 3
        rho = 0.9

        if x.sum() == 0:
            return 1

        if len(self.history) == 0:
            return np.power(rho, max(0, x.sum() - tau + 1))

        history = np.stack(self.history)[::-1]
        past = history.sum(0)
        xb = x.astype(bool)
        is_new = xb & ~past.astype(bool)
        distance = 1 + (history * xb).argmax(0)
        too_old = distance >= tau

        penalized = max(0, (too_old | is_new).sum() - tau + 1)
        return np.power(rho, penalized)

    def joint(self, prior, utt):
        # p(r,s | u)
        # prior: num_configs * 7
        # p(r=0|u,s)p(s)
        # = \sum_z p(s)p(z|s) p(r=0|u,s)p(r=0|u,z)
        # = p(s)p(r=0|u,s)\sum_z p(z|s)p(r=0|u,z)
        # = p(s)(1 - p(r=1|u,s)) |z|C|u|9^-|u|

        u = int(utt.sum())
        utt_idx = np.right_shift(np.packbits(utt), 8-self.num_dots).item()
        temporal_confirms = np.array([
            self.temporal_confirm(x, self.history) for x in self.configs
        ])
        tc = temporal_confirms[utt_idx] if self.use_temporal else 1
        sd = self.spatial_denies[utt_idx] if self.use_spatial else 0

        p_r1 = []
        p_r0 = []
        for s,ps in enumerate(prior):
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            c_likelihood = self.config_likelihood[utt_idx,s].item()
            likelihood = c_likelihood * tc * (1-sd)
            #if len(self.history) > 0:
                #import pdb; pdb.set_trace()
            #for i,d in enumerate(utt):
                #if d == 1:
                    #likelihood *= self.likelihood[1,d,state_config[i]]
            distractor_prob = 1 - comb(z,u) * 9. ** (-u)

            #p_deny = (1 - likelihood)*distractor_prob
            p_deny = (1 - likelihood)*distractor_prob
            #if utt.sum() > 1 and state_config.sum() > 4:
                #print(p_deny)
                #if self.spatial_denies[s] < .1 and c_likelihood > 0.9:
                    #import pdb; pdb.set_trace()

            #print(p_deny)
            if p_deny > 1:
                print(likelihood)
                print(distractor_prob)
                print(td)
                print(sd)
                import pdb; pdb.set_trace()

            p_r0.append(p_deny * ps)
            p_r1.append((1 - p_deny) * ps)
            #if (utt == np.array([0, 1, 0, 0, 1, 0, 0])).all() and (state_config == np.array([0,1,0,1,1,0,0])).all():
            #if (utt == np.array([0, 1, 0, 1, 0, 0, 0])).all() and (state_config == np.array([0,1,0,1,1,0,0])).all():
                #import pdb; pdb.set_trace()
        return np.array((p_r0, p_r1))

class EgoCostBelief(CostBelief):
    # ABLATED version of CostBelief
    # Does not consider unshared dots
    # same method as belief.py:ConfigBelief
    def joint(self, prior, utt):
        # p(r | u,s)
        # prior: num_configs * 7
        # p(r=0|u,s)p(s)
        # = \sum_z p(s)p(z|s) p(r=0|u,s)p(r=0|u,z)
        # = p(s)p(r=0|u,s)\sum_z p(z|s)p(r=0|u,z)
        # = p(s)(1 - p(r=1|u,s)) |z|C|u|9^-|u|
        p_r1 = []
        p_r0 = []
        for s,ps in enumerate(prior):
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            u = int(utt.sum())
            utt_idx = np.right_shift(np.packbits(utt), 8-self.num_dots)
            likelihood = self.config_likelihood[utt_idx,s].item()
            #for i,d in enumerate(utt):
                #if d == 1:
                    #likelihood *= self.likelihood[1,d,state_config[i]]
            #distractor_prob = 1 - comb(z,u) * 9. ** (-u)
            p_r1.append(likelihood * ps)
            p_r0.append((1 - likelihood) * ps)
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
    #belief = CostBelief(num_dots, ctx, num_size_buckets=5, num_color_buckets=5, use_temporal=False)
    #belief = OrBelief(num_dots, ctx, num_size_buckets=5, num_color_buckets=5)
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
    #fig, ax = plt.subplots(N, figsize=(3, 3*N))

    prior = belief.prior
    for n in range(N):
        EdHs = belief.compute_EdHs(prior)
        mEdHs = belief.compute_marginal_EdHs(prior).max(-1)
        # looks like the marginal posterior wants to improve the probability
        # of one dot by asking about ALL OTHERS
        #mps = belief.compute_marginal_posteriors(prior)

        #costs = belief.compute_processing_costs(new_weight, distance_weight)
        #costs = costs / 25 / 7
        #costs = costs / belief.num_dots / 2
        #utt = belief.configs[(mEdHs + costs).argmax()]
        #utt = belief.configs[(EdHs + costs).argmax()]
        utt = belief.configs[EdHs.argmax()]
        #utt = belief.configs[mEdHs.argmax()]
        #print(belief.configs[(mEdHs + costs).argmax()])
        #print(belief.configs[(EdHs + costs).argmax()])
        print("utt", belief.configs[EdHs.argmax()])
        print("marg utt", belief.configs[mEdHs.argmax()])

        uttb = utt.astype(bool)
        fig, ax = plt.subplots()
        ax.scatter(
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
        ax.scatter(xy[uttb,0], xy[uttb,1], marker="x", s=100, c="r")
        for i, id in enumerate(ids):
            ax.annotate(id, (xy[i,0]+.025, xy[i,1]+.025))
        plt.show()
        plt.clf()

        response = int(input())
        print("prior", belief.marginals(prior))
        #new_prior = belief.posterior(prior, utt, 1 if n % 2 == 0 else 0)
        #new_prior = belief.posterior(prior, utt, 1)
        #new_prior = belief.posterior(prior, utt, 0)
        new_prior = belief.posterior(prior, utt, response)
        print("posterior", belief.marginals(new_prior))
        #import pdb; pdb.set_trace()

        belief.history.append(utt)
        prior = new_prior

        #import pdb; pdb.set_trace()
    # automate logging based on isinstance(belief, CostBelief)
    #plt.show()
    #plt.savefig("dbg_plots/all_yes.png")
    #plt.savefig("dbg_plots/all_yes_baseline.png")
    #plt.savefig("dbg_plots/all_no.png")
    #plt.savefig("dbg_plots/all_no_baseline.png")

    utt = np.array([1,0,0,1,0,1,1])
    uttb = utt.astype(bool)
    a = belief.spatial_deny(utt, ctx)
    b = belief.temporal_deny(utt, belief.history)

    #import pdb; pdb.set_trace()

