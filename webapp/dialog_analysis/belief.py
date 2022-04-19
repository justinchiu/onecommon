import numpy as np
from scipy.special import logsumexp as lse

#random.seed(1234)
#np.random.seed(1234)

# discrete entropy
def entropy(px):
    Hx = np.where(px > 0, px * np.log(px), 0)
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

    def p_response(self, prior, ask):
        raise NotImplementedError

    def posterior(self, prior, ask, response):
        raise NotImplementedError

    def info_gain(self, prior, ask, response):
        Hs = entropy(prior)
        Hs_r = entropy(self.posterior(prior, ask, response))
        return Hs - Hs_r

    def expected_info_gain(self, prior, ask):
        raise NotImplementedError


class IndependentBelief(Belief):
    def __init__(self, num_dots):
        super().__init__(num_dots)
        # VERY PESSIMISTIC PRIOR
        # actually, this is incorrect.
        # if we know 6/7 overlap, the marginal dist should be 1/7 not included
        # given K overlap, marginal dist is 1 - 6Ck / 7Ck = k/7
        # guess we should not assume K overlap though, be even dumber?
        state_prior = np.ones((num_dots,)) / 2
        self.prior= np.stack((state_prior, 1-state_prior), 1)

        # initialize basic likelihood
        correct = 0.9
        error = 1 - correct
        likelihood = np.ones((2,2,2)) * error
        # ask about something, get correct answer
        likelihood[1,1,1] = correct
        likelihood[0,1,0] = correct
        # if you dont ask about something, no change
        likelihood[:,0] = 1
        self.likelihood = likelihood

    # RESPONSE IS FOR ALL DOTS INDEPENDENTLY
    def p_response(self, prior, ask):
        return (self.likelihood[:,ask] * prior).sum(-1).T

    def posterior(self, prior, ask, response):
        f = self.likelihood[response, ask] * prior
        return f / f.sum(-1, keepdims=True)

    def info_gain(self, prior, ask, response):
        Hs = entropy(prior)
        Hs_r = entropy(self.posterior(prior, ask, response))
        return (Hs - Hs_r)[ask.astype(bool)].sum()

    def expected_info_gain(self, prior, ask):
        p_response = self.p_response(prior, ask)
        Hs = entropy(prior)
        r0 = np.zeros((self.num_dots,), dtype=int)
        r1 = np.ones((self.num_dots,), dtype=int)
        Hs_r0 = entropy(self.posterior(prior, ask, r0))
        Hs_r1 = entropy(self.posterior(prior, ask, r1))
        EHs_r = (p_response * np.stack((Hs_r0, Hs_r1), 1)).sum(-1)
        return (Hs - EHs_r)[ask.astype(bool)].sum()



class JointBelief(Belief):
    """ RESPONSES ARE IN PARALLEL, SO ASK ABOUT ALL 7 DOTS
        RESPONSE IS FOR ALL DOTS JOINTLY
        ergo state prior/posterior are of size 2^7
        variational approximation to follow
    """

    def __init__(self, num_dots, overlap_size=None):
        super().__init__(num_dots, overlap_size)

        self.prior = np.ones((2 ** num_dots,))
        if overlap_size is not None:
            self.prior[self.configs.sum(-1) < overlap_size] = 0
        self.prior = self.prior / self.prior.sum()

        # initialize basic likelihood
        correct = 0.9
        error = 1 - correct
        likelihood = np.ones((2,2,2)) * error
        # ask about something, get correct answer
        likelihood[1,1,1] = correct
        likelihood[0,1,0] = correct
        # if you dont ask about something, no change
        likelihood[:,0] = 1
        self.likelihood = likelihood

    def p_response(self, prior, ask):
        # prior: num_configs * 7
        # \sum_s p(r=1 | u, s)p(s) = \sum_s \prod_i p(r=1 | ui, si)p(s)
        p_r1 = 0
        p_r0 = 0
        for s,p in enumerate(prior):
            likelihood = 1
            for i,d in enumerate(ask):
                likelihood *= self.likelihood[1,d,self.configs[s,i]]
            p_r1 += likelihood * p
            p_r0 += (1-likelihood) * p
        #p_r0 = 1 - p_r1
        return np.array((p_r0, p_r1))

    def posterior(self, prior, ask, response):
        # p(r=., s | u) = \prod_i p(r=. | ui, si)p(s)
        p_r0s_u = []
        p_r1s_u = []
        for s,p in enumerate(prior):
            likelihood = 1
            for i,d in enumerate(ask):
                likelihood *= self.likelihood[1,d,self.configs[s,i]]
            p_r1s_u.append(likelihood * p)
            p_r0s_u.append((1-likelihood) * p)
        p_r1s_u = np.array(p_r1s_u)
        p_r0s_u = np.array(p_r0s_u)
        p_s_ur1 = p_r1s_u / p_r1s_u.sum(-1, keepdims=True)
        p_s_ur0 = p_r0s_u / p_r0s_u.sum(-1, keepdims=True)
        return p_s_ur1 if response == 1 else p_s_ur0

    def info_gain(self, prior, ask, response):
        Hs = entropy(prior)
        Hs_r = entropy(self.posterior(prior, ask, response))
        return Hs - Hs_r

    def expected_info_gain(self, prior, ask):
        p_response = self.p_response(prior, ask)
        Hs = entropy(prior)
        Hs_r0 = entropy(self.posterior(prior, ask, 0))
        Hs_r1 = entropy(self.posterior(prior, ask, 1))
        EHs_r = (p_response * np.array((Hs_r0, Hs_r1))).sum()
        return Hs - EHs_r

num_dots = 7

ask = np.array([1 if x in [2,5] else 0 for x in range(num_dots)])
response = np.array([1 if x in [2,5] else 0 for x in range(num_dots)])
configs = np.array([
    np.unpackbits(np.array([x], dtype=np.ubyte))[8 - num_dots:]
    for x in range(2 ** num_dots)
])

# refactor later into test
belief = IndependentBelief(num_dots)
p_s_ar = belief.posterior(belief.prior, ask, response)
dH = belief.info_gain(belief.prior, ask, response)

EdH = belief.expected_info_gain(belief.prior, ask)

EdHs = []
for utt in configs[1:]:
    EdH = belief.expected_info_gain(belief.prior, utt)
    EdHs.append(EdH)
EdHs = np.array(EdHs)
print(EdHs)

# joint model
belief = JointBelief(num_dots, overlap_size = 4)
response = 1
p_s_ar = belief.posterior(belief.prior, ask, response)
dH = belief.info_gain(belief.prior, ask, response)

EdH = belief.expected_info_gain(belief.prior, ask)

EdHs = []
for utt in configs:
    p_r = belief.p_response(belief.prior, utt)
    EdH = belief.expected_info_gain(belief.prior, utt)
    EdHs.append(EdH)
EdHs = np.array(EdHs)
print(EdHs)
#print(belief.configs[1:])
print(EdHs.max(), EdHs.argmax(), configs[EdHs.argmax()])

import pdb; pdb.set_trace()

