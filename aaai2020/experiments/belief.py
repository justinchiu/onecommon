
import math

import itertools
import numpy as np
from scipy.special import logsumexp as lse
from scipy.special import comb

#random.seed(1234)
#np.random.seed(1234)

def process_ctx(ctx):
    ctx = np.array(ctx, dtype=float).reshape(-1, 4)
    # ctx: [x, y, size, color]
    num_buckets = 4
    min_ = ctx.min(0)
    max_ = ctx.max(0)
    min_size, max_size = min_[2], max_[2]
    min_color, max_color = min_[3], max_[3]
    size_buckets = np.linspace(min_size, max_size, num_buckets)
    color_buckets = np.linspace(min_color, max_color, num_buckets)
    sizes = ctx[:,2]
    colors = ctx[:,3]

    size_idxs = (size_buckets[:-1,None] <= sizes) & (sizes <= size_buckets[1:,None])
    color_idxs = (color_buckets[:-1,None] <= colors) & (colors <= color_buckets[1:,None])
    return np.stack((size_idxs.T.nonzero()[1], color_idxs.T.nonzero()[1]), 1)

def safe_log(x, eps=1e-10):
    result = np.where(x > eps, x, 0)
    np.log(result, out=result, where=result > 0)
    return result

# discrete entropy
def entropy(px):
    Hx = px * safe_log(px)
    return -(Hx).sum(-1)

class Dot:
    def __init__(self, item):
        for k,v in item.items():
            setattr(self, k, v)

    def html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size
        f = self.color
        label = f'<text x="{x+12}" y="{y-12}" font-size="18">{self.id}</text>'
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="{f}" /> {label}'

    def select_html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size + 2
        f = self.color # ignored
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="red" stroke-width="3" stroke-dasharray="3,3"  />'

    def intersect_html(self, shift=0):
        x = self.x + shift
        y = self.y
        r = self.size + 4
        f = self.color # ignored
        return f'<circle cx="{x}" cy="{y}" r="{r}" fill="none" stroke="blue" stroke-width="3" stroke-dasharray="3,3"  />'

    def __repr__(self):
        return f"Dot {self.id}: ({self.x}, {self.y}) r={self.size} f={self.color}"


class Belief:
    def __init__(self, num_dots, overlap_size=None):
        self.num_dots = num_dots
        self.overlap_size = overlap_size
        self.configs = np.array([
            np.unpackbits(np.array([x], dtype=np.ubyte))[8-num_dots:]
            for x in range(2 ** num_dots)
        ])
        self.num_configs = 2 ** num_dots

    def joint(self, prior, utt):
        raise NotImplementedError

    def p_response(self, prior, utt):
        raise NotImplementedError

    def posterior(self, prior, utt, response):
        raise NotImplementedError

    def info_gain(self, prior, utt, response):
        Hs = entropy(prior)
        Hs_r = entropy(self.posterior(prior, utt, response))
        return Hs - Hs_r

    def expected_info_gain(self, prior, utt):
        raise NotImplementedError

    def compute_EdHs(self, prior):
        EdHs = []
        for utt in self.configs:
            EdH = self.expected_info_gain(prior, utt)
            EdHs.append(EdH)
        return np.array(EdHs)

    def viz_belief(self, p, n=5):
        # decreasing order
        idxs = (-p).argsort()[:n]
        cs = self.configs[idxs]
        ps = p[idxs]
        return cs, ps

    def marginals(self, p):
        return (self.configs * p[:,None]).sum(0)


class IndependentBelief(Belief):
    """
    Fully independent partner model
    * response r: num_dots
    * utterance u: num_dots
    * state s: num_dots
    p(r|u,s) = prod_i p(r_i|u_i,s_i)
    Underestimates failures from large configurations due to
    independence assumption.
    """
    def __init__(self, num_dots, correct=0.9):
        super().__init__(num_dots)
        # VERY PESSIMISTIC PRIOR
        # actually, this is incorrect.
        # if we know 6/7 overlap, the marginal dist should be 1/7 not included
        # given K overlap, marginal dist is 1 - 6Ck / 7Ck = k/7
        # guess we should not assume K overlap though, be even dumber?
        state_prior = np.ones((num_dots,)) / 2
        self.prior= np.stack((state_prior, 1-state_prior), 1)

        # initialize basic likelihood
        error = 1 - correct
        likelihood = np.ones((2,2,2)) * error
        # utt about something, get correct answer
        likelihood[1,1,1] = correct
        likelihood[0,1,0] = correct
        # if you dont utt about something, no change
        likelihood[:,0] = 1
        self.likelihood = likelihood

    # RESPONSE IS FOR ALL DOTS INDEPENDENTLY
    def p_response(self, prior, utt):
        return (self.likelihood[:,utt] * prior).sum(-1).T

    def posterior(self, prior, utt, response):
        f = self.likelihood[response, utt] * prior
        return f / f.sum(-1, keepdims=True)

    def info_gain(self, prior, utt, response):
        Hs = entropy(prior)
        Hs_r = entropy(self.posterior(prior, utt, response))
        return (Hs - Hs_r)[utt.astype(bool)].sum()

    def expected_info_gain(self, prior, utt):
        p_response = self.p_response(prior, utt)
        Hs = entropy(prior)
        r0 = np.zeros((self.num_dots,), dtype=int)
        r1 = np.ones((self.num_dots,), dtype=int)
        Hs_r0 = entropy(self.posterior(prior, utt, r0))
        Hs_r1 = entropy(self.posterior(prior, utt, r1))
        EHs_r = (p_response * np.stack((Hs_r0, Hs_r1), 1)).sum(-1)
        return (Hs - EHs_r)[utt.astype(bool)].sum()



class AndBelief(Belief):
    """
    Noisy-and model for response modeling.
    Partner will (noisily) confirm an utterance if they see all dots mentioned.
    * response r: 1
    * utterance u: num_dots
    * state s: num_dots
    p(r=1|u,s) = prod_i p(r=1|u_i=1,s_i)
    Accurately estimates failure of large configurations,
    under-estimates failure of small configurations due to ignoring partial observability.
    """

    def __init__(self, num_dots, overlap_size=None, correct=0.9):
        super().__init__(num_dots, overlap_size)

        self.prior = np.ones((2 ** num_dots,))
        if overlap_size is not None:
            self.prior[self.configs.sum(-1) < overlap_size] = 0
            #self.prior[self.configs.sum(-1) != overlap_size] = 0
            self.prior[-1] = 0
        self.prior = self.prior / self.prior.sum()

        # initialize basic likelihood
        error = 1 - correct
        likelihood = np.ones((2,2,2)) * error
        # utt about something, get correct answer
        likelihood[1,1,1] = correct
        likelihood[0,1,0] = correct
        # if you dont utt about something, no change
        likelihood[:,0] = 1
        self.likelihood = likelihood

    def p_response(self, prior, utt):
        # prior: num_configs * 7
        # \sum_s p(r=1 | u, s)p(s) = \sum_s \prod_i p(r=1 | ui, si)p(s)
        p_r1 = 0
        p_r0 = 0
        for s,p in enumerate(prior):
            likelihood = 1
            for i,d in enumerate(utt):
                likelihood *= self.likelihood[1,d,self.configs[s,i]]
            p_r1 += likelihood * p
            p_r0 += (1-likelihood) * p
        #p_r0 = 1 - p_r1 # this is equivalent
        return np.array((p_r0, p_r1))

    def posterior(self, prior, utt, response):
        # p(r=., s | u) = \prod_i p(r=. | ui, si)p(s)
        p_r0s_u = []
        p_r1s_u = []
        for s,p in enumerate(prior):
            likelihood = 1
            for i,d in enumerate(utt):
                likelihood *= self.likelihood[1,d,self.configs[s,i]]
            p_r1s_u.append(likelihood * p)
            p_r0s_u.append((1-likelihood) * p)
        p_r1s_u = np.array(p_r1s_u)
        p_r0s_u = np.array(p_r0s_u)
        Z1 = p_r1s_u.sum(-1, keepdims=True)
        p_s_ur1 = p_r1s_u / Z1 if Z1 > 0 else np.ones((2 ** self.num_dots,)) / 2 ** self.num_dots
        Z2 = p_r0s_u.sum(-1, keepdims=True)
        p_s_ur0 = p_r0s_u / Z2 if Z2 > 0 else np.ones((2 ** self.num_dots,)) / 2 ** self.num_dots
        return p_s_ur1 if response == 1 else p_s_ur0

    def info_gain(self, prior, utt, response):
        Hs = entropy(prior)
        Hs_r = entropy(self.posterior(prior, utt, response))
        return Hs - Hs_r

    def expected_info_gain(self, prior, utt):
        p_response = self.p_response(prior, utt)
        Hs = entropy(prior)
        Hs_r0 = entropy(self.posterior(prior, utt, 0))
        Hs_r1 = entropy(self.posterior(prior, utt, 1))
        EHs_r = (p_response * np.array((Hs_r0, Hs_r1))).sum()
        return Hs - EHs_r

class AndOrBelief(AndBelief):
    """
    And-or model for response modeling.
    Partner will (noisily) confirm an utterance if they see all dots mentioned
    OR have matching dots in unobserved context.
    The OR happens at the dot-level.
    * response r: 1
    * utterance u: num_dots
    * state s: num_dots
    * unobserved partner dots z: num_dots - |s|
    p(r=1|u,s) = prod_i p(r=1|u_i,s_i,z) = prod_i 1 - p(r=0|ui,si)p(r=0|ui,z)
    Accurately estimates failure of small and large configurations.
    As the OR happens at the dot level, does not prefer large configurations.

    Note on p(r=0|ui,z) = (8/9)^|z|:
        color = light, medium, dark
        size = small, medium, dark
        Assume descriptions are all independent, so only 9 possibilities
        for each dot in z
        Size of z: remaining dots outside of s |z| = num_dots - |s|
    """
    def p_response(self, prior, utt):
        # prior: num_configs * 7
        # \sum_s p(r=1 | u, s)p(s)
        # = \sum_s,z p(s)p(z|s) \prod_i 1-p(r=0|ui,si)p(r=0|ui,z)
        # = \sum_s p(s) \prod_i 1-p(r=0|ui,si)(8/9)^{n-|s|}
        p_r1 = 0
        p_r0 = 0
        for s,ps in enumerate(prior):
            likelihood = 1
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            for i,d in enumerate(utt):
                if d == 1:
                    disconfirm = self.likelihood[0,d,state_config[i]] * (8/9) ** z
                    likelihood *= 1 - disconfirm
            p_r1 += likelihood * ps
            p_r0 += (1-likelihood) * ps
        #p_r0 = 1 - p_r1 # this is equivalent
        return np.array((p_r0, p_r1))

    def posterior(self, prior, utt, response):
        # p(r=., s | u) = \prod_i p(r=. | ui, si)p(s)
        p_r0s_u = []
        p_r1s_u = []
        for s,p in enumerate(prior):
            likelihood = 1
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            for i,d in enumerate(utt):
                if d == 1:
                    disconfirm = self.likelihood[0,d,state_config[i]] * (8/9) ** z
                    likelihood *= 1 - disconfirm
            p_r1s_u.append(likelihood * p)
            p_r0s_u.append((1-likelihood) * p)
        p_r1s_u = np.array(p_r1s_u)
        p_r0s_u = np.array(p_r0s_u)
        Z1 = p_r1s_u.sum(-1, keepdims=True)
        p_s_ur1 = p_r1s_u / Z1 if Z1 > 0 else np.ones((2 ** self.num_dots,)) / 2 ** self.num_dots
        Z0 = p_r0s_u.sum(-1, keepdims=True)
        p_s_ur0 = p_r0s_u / Z0 if Z0 > 0 else np.ones((2 ** self.num_dots,)) / 2 ** self.num_dots
        return p_s_ur1 if response == 1 else p_s_ur0

class OrAndBelief(AndBelief):
    """
    Or-and model for response modeling.
    Partner will (noisily) confirm an utterance if they see all dots mentioned
    OR have matching dots in unobserved context.
    The OR happens at the config level.
    * response r: 1
    * utterance u: num_dots
    * state s: num_dots
    * unobserved partner dots z: num_dots - |s|

    Noisy-AND for dots and state
        p(r=1|u,s) = prod_i p(r=1|u_i,s_i)
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

    def joint(self, prior, utt):
        # p(r | u,s)
        # prior: num_configs * 7
        # p(r=0|u,s)p(s)
        # = \sum_z p(s)p(z|s) p(r=0|u,s)p(r=0|u,z)
        # = p(s)p(r=0|u,s)\sum_z p(z|s)p(r=0|u,z)
        # = p(s)(1-\prod_i p(r=1|ui,si)) \sum_z p(z|s)p(r=0|u,z)
        # = p(s)(1-\prod_i p(r=1|ui,si)) |z|C|u|9^-|u|
        p_r1 = []
        p_r0 = []
        for s,ps in enumerate(prior):
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            u = int(utt.sum())
            likelihood = 1
            for i,d in enumerate(utt):
                if d == 1:
                    likelihood *= self.likelihood[1,d,state_config[i]]
            distractor_prob = 1 - comb(z,u) * 9. ** (-u)
            p_r0.append((1 - likelihood)*distractor_prob * ps)
            p_r1.append((1- (1-likelihood)*distractor_prob) * ps)
        return np.array((p_r0, p_r1))

    def p_response(self, prior, utt):
        return self.joint(prior, utt).sum(1)

    def posterior(self, prior, utt, response):
        # p(r=., s | u) = \prod_i p(r=. | ui, si)p(s)
        p_rs_u = self.joint(prior, utt)
        Z = p_rs_u.sum(1, keepdims=True)
        unif = np.ones((2, 2 ** self.num_dots)) / 2 ** self.num_dots
        p_s_ur = np.divide(p_rs_u, Z, out=unif, where=Z>0)
        return p_s_ur[response]

class OrBelief(OrAndBelief):
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
    def __init__(self, num_dots, ctx, correct=0.95, overlap_size=None,):
        super().__init__(num_dots, overlap_size=overlap_size, correct=correct)
        self.ctx = ctx
        self.size_color = process_ctx(ctx)
        self.xy = ctx[:,:2]
        # initialize config_likelihood based on configuration resolution
        self.config_likelihood = np.zeros((self.num_configs), dtype=float)
        for u, utt in enumerate(self.configs):
            self.config_likelihood[u] = correct if self.can_resolve_utt(utt) else 1 - correct

    def can_resolve_utt(self, utt):
        size_color = self.size_color
        xy = self.xy

        utt_size = utt.sum().item()

        arange = np.arange(self.num_dots)
        perms = np.array(list(itertools.permutations(np.arange(utt_size))))
        num_perms = perms.shape[0]
        pairwise_combs = np.array(list(itertools.combinations(np.arange(utt_size), 2)))

        utt_sc = size_color[utt.astype(bool)]
        utt_xy = xy[utt.astype(bool)]
        utt_pairwise_xy = utt_xy[pairwise_combs]
        utt_rel_xy = utt_pairwise_xy[:,0] > utt_pairwise_xy[:,1]

        for config in self.configs:
            # get all dot combinations of size utt_size
            config_ids = arange[config.astype(bool)]
            combs = np.array(list(itertools.combinations(config_ids, utt_size)))
            for comb in combs:
                comb_sc = size_color[comb]
                comb_xy = xy[comb]

                # this checks if all dots have a match, just to filter?
                sc_match = (utt_sc[:,None] == comb_sc[None,:]).all(-1).any(1).all()

                if sc_match:
                    # check if there is a permutation that matches
                    sc_perms = comb_sc[perms]
                    xy_perms = comb_xy[perms]

                    sc_matches = (utt_sc == sc_perms).reshape(num_perms, -1).all(-1)
                    xy_potential_matches = xy_perms[sc_matches]
                    for xy_config in xy_potential_matches:
                        config_pairwise_xy = xy_config[pairwise_combs]
                        config_rel_xy = config_pairwise_xy[:,0] > config_pairwise_xy[:,1]

                        xy_match = (utt_rel_xy == config_rel_xy).all()
                        if xy_match:
                            return True
        return False

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
            likelihood = self.config_likelihood[utt_idx].item()
            for i,d in enumerate(utt):
                if d == 1:
                    likelihood *= self.likelihood[1,d,state_config[i]]
            distractor_prob = 1 - comb(z,u) * 9. ** (-u)
            p_r0.append((1 - likelihood)*distractor_prob * ps)
            p_r1.append((1- (1-likelihood)*distractor_prob) * ps)
        return np.array((p_r0, p_r1))

class OrAndOrBelief(OrAndBelief):
    """
    Or-and-or model for response modeling.
    Partner will (noisily) confirm an utterance if they see all dots mentioned
    OR have matching dots in unobserved context.
    The OR happens at the config level.
    * response r: {0,1}
    * utterance u: num_dots
    * state s: num_dots
    * unobserved partner dots z: num_dots - |s|

    OR at the config for state / distractors
        p(r=0|u,s,z) = 1-p(r=0|u,s)p(r=0|u,z)
    Dot distractors
        p(r=0|u,z) = 1 - |z|C|u| 9^-|u|
    AND for dots and state
        p(r=1|u,s) = prod_i p(r=1|u_i,s)
        p(r=0|u,s) = 1-p(r=1|u,s)
    Or for each dot
        p(r=1|u_i,s) = 0.9 if any s_j == u_i

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
        dots,
        correct=0.9,
        overlap_size=None,
    ):
        super().__init__(num_dots, overlap_size=overlap_size, correct=correct)

        self.dots = dots

        # initialize likelihood
        error = 1 - correct
        likelihood = np.ones((2,num_dots,2**num_dots)) * error

        for c, config in enumerate(self.configs):
            # utt about something, get correct answer
            # if bucketed attributes match
            for i, dot in enumerate(dots):
                if dot in dots[config.astype(bool)]:
                    likelihood[1,i,c] = correct
                else:
                    likelihood[0,i,c] = correct
        self.likelihood = likelihood

    def joint(self, prior, utt):
        # p(r | u,s)
        # prior: num_configs * 7
        # p(r=0|u,s)p(s)
        # = \sum_z p(s)p(z|s) p(r=0|u,s)p(r=0|u,z)
        # = p(s)p(r=0|u,s)\sum_z p(z|s)p(r=0|u,z)
        # = p(s)p(r=0|u,s) |z|C|u|9^-|u|
        p_r1 = []
        p_r0 = []
        for s,ps in enumerate(prior):
            likelihood = 1
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            u = int(utt.sum())
            #utt_idx = np.right_shift(np.packbits(utt), 8-self.num_dots)
            likelihood = 1
            for i,d in enumerate(utt):
                if d == 1:
                    likelihood *= self.likelihood[1,i,s]
            distractor_prob = 1 - comb(z,u) * 9. ** (-u)
            p_r0.append((1 - likelihood)*distractor_prob * ps)
            p_r1.append((1- (1-likelihood)*distractor_prob) * ps)
        return np.array((p_r0, p_r1))



if __name__ == "__main__":
    np.seterr(all="raise")

    num_dots = 7
    overlap_size = 4
    #num_dots = 4
    #overlap_size = None

    utt = np.array([1 if x in [2,5] else 0 for x in range(num_dots)])
    response = np.array([1 if x in [2,5] else 0 for x in range(num_dots)])
    configs = np.array([
        np.unpackbits(np.array([x], dtype=np.ubyte))[8 - num_dots:]
        for x in range(2 ** num_dots)
    ])

    # refactor later into test
    print("IND MODEL")
    belief = IndependentBelief(num_dots)
    EdHs = belief.compute_EdHs(belief.prior)
    cs, hs = belief.viz_belief(EdHs)
    print(cs)
    print(hs)

    # joint model
    print("JOINT MODEL")
    belief = AndBelief(num_dots, overlap_size = overlap_size)
    EdHs = belief.compute_EdHs(belief.prior)
    cs, hs = belief.viz_belief(EdHs)
    print(cs)
    print(hs)

    # po dot model
    print("PO DOT MODEL")
    belief = AndOrBelief(num_dots, overlap_size = overlap_size)
    EdHs = belief.compute_EdHs(belief.prior)
    cs, hs = belief.viz_belief(EdHs)
    print(cs)
    print(hs)

    # po config model
    print("PO CONFIG MODEL")
    belief = OrAndBelief(num_dots, overlap_size = overlap_size)
    EdHs = belief.compute_EdHs(belief.prior)
    cs, hs = belief.viz_belief(EdHs)
    print(cs)
    print(hs)

    print("construct prior that elicits 3-mention")
    belief = OrAndBelief(num_dots, overlap_size = None)
    a = [1,1,0,0,0,0,0]
    b = [1,0,0,1,0,0,0]
    c = [1,1,1,0,0,0,0]
    a_idx = (configs == a).all(1).nonzero()[0]
    b_idx = (configs == b).all(1).nonzero()[0]
    c_idx = (configs == c).all(1).nonzero()[0]
    prior = np.ones_like(belief.prior)
    prior[a_idx] = 50
    prior[b_idx] = 50
    prior[c_idx] = 50
    prior = prior / prior.sum()
    EdHs = belief.compute_EdHs(prior)
    cs, hs = belief.viz_belief(EdHs)
    print(cs)
    print(hs)

    print("give conflicting responses for single dot")
    belief = OrAndBelief(num_dots, overlap_size = None)
    u = np.array([1,0,0,0,0,0,0])
    prior = belief.posterior(belief.prior,u,1)
    print("marginals after positive")
    print(belief.marginals(prior))
    prior = belief.posterior(prior,u,0)
    print("marginals after negative")
    print(belief.marginals(prior))
    prior = belief.posterior(prior,u,0)
    print("marginals after second negative")
    print(belief.marginals(prior))
    EdHs = belief.compute_EdHs(prior)
    cs, ps = belief.viz_belief(prior)
    print(cs)
    print(ps)

    print("give conflicting responses for multi dot")
    belief = OrAndBelief(num_dots, overlap_size = None)
    u = np.array([1,1,0,0,0,0,0])
    prior = belief.posterior(belief.prior,u,1)
    print("marginals after positive")
    print(belief.marginals(prior))
    prior = belief.posterior(prior,u,0)
    print("marginals after negative")
    print(belief.marginals(prior))
    prior = belief.posterior(prior,u,0)
    print("marginals after second negative")
    print(belief.marginals(prior))
    EdHs = belief.compute_EdHs(prior)
    cs, ps = belief.viz_belief(prior)
    print(cs)
    print(ps)

    print("20 questions simulation")

    if overlap_size is not None:
        intersect_size = math.ceil(num_dots / 2)
    else:
        intersect_size = overlap_size
    state = np.random.gumbel(size=num_dots)
    idxs1 = state.argsort()[:intersect_size]
    idxs0 = state.argsort()[intersect_size:]
    state[idxs1] = 1
    state[idxs0] = 0

    """
    print("true state")
    print(state)

    prior = belief.prior
    for t in range(10):
        EdHs = belief.compute_EdHs(prior)
        best_idx = EdHs.argmax()
        next_utt = configs[best_idx]
        response = int(state.astype(bool)[next_utt.astype(bool)].all())
        posterior = belief.posterior(prior, next_utt, response)
        print(f"utt {next_utt}: response {response}")
        print("posterior")
        cs, ps = belief.viz_belief(posterior)
        print(cs)
        print(ps)
        prior = posterior
    """

    print("spatial context tests")
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

    # goes in init
    size_color = process_ctx(ctx)
    xy = ctx[:,:2]

    utt = np.array([1,0,1,1,0,0,0])


                    import pdb; pdb.set_trace()
