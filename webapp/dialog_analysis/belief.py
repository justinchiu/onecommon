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
    """
    Fully independent partner model
    * response r: num_dots
    * utterance u: num_dots
    * state s: num_dots
    p(r|u,s) = prod_i p(r_i|u_i,s_i)
    Underestimates failures from large configurations due to
    independence assumption.
    """
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



class AndBelief(Belief):
    """
    Noisy-and model for response modeling.
    Partner will (noisily) confirm an utterance if they see all dots mentioned.
    * response r: 1
    * utterance u: num_dots
    * state s: num_dots
    p(r=1|u,s) = prod_i p(r=1|u_i,s_i)
    Accurately estimates failure of large configurations,
    under-estimates failure of small configurations due to ignoring partial observability.
    """

    def __init__(self, num_dots, overlap_size=None):
        super().__init__(num_dots, overlap_size)

        self.prior = np.ones((2 ** num_dots,))
        if overlap_size is not None:
            self.prior[self.configs.sum(-1) < overlap_size] = 0
            #self.prior[self.configs.sum(-1) != overlap_size] = 0
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
        #p_r0 = 1 - p_r1 # this is equivalent
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

class AndOrBelief(AndBelief):
    """
    Noisy-and-or model for response modeling.
    Partner will (noisily) confirm an utterance if they see all dots mentioned
    OR have matching dots in unobserved context.
    * response r: 1
    * utterance u: num_dots
    * state s: num_dots
    * unobserved partner dots z: num_dots - |s|
    p(r=1|u,s) = prod_i p(r=1|u_i,s_i,z) = prod_i 1 - p(r=0|ui,si)p(r=0|ui,z)
    Accurately estimates failure of small and large configurations.

    Note on p(r=0|ui,z) = (8/9)^|z|:
        color = light, medium, dark
        size = small, medium, dark
        Assume descriptions are all independent, so only 9 possibilities
        for each dot in z
        Size of z: remaining dots outside of s |z| = num_dots - |s|
    """
    def p_response(self, prior, ask):
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
            for i,d in enumerate(ask):
                if d == 1:
                    disconfirm = self.likelihood[0,d,state_config[i]] * (8/9) ** z
                    likelihood *= 1 - disconfirm
            p_r1 += likelihood * ps
            p_r0 += (1-likelihood) * ps
        #p_r0 = 1 - p_r1 # this is equivalent
        return np.array((p_r0, p_r1))

    def posterior(self, prior, ask, response):
        # p(r=., s | u) = \prod_i p(r=. | ui, si)p(s)
        p_r0s_u = []
        p_r1s_u = []
        for s,p in enumerate(prior):
            likelihood = 1
            state_config = self.configs[s]
            z = self.num_dots - state_config.sum()
            for i,d in enumerate(ask):
                if d == 1:
                    disconfirm = self.likelihood[0,d,state_config[i]] * (8/9) ** z
                    likelihood *= 1 - disconfirm
            p_r1s_u.append(likelihood * p)
            p_r0s_u.append((1-likelihood) * p)
        p_r1s_u = np.array(p_r1s_u)
        p_r0s_u = np.array(p_r0s_u)
        p_s_ur1 = p_r1s_u / p_r1s_u.sum(-1, keepdims=True)
        p_s_ur0 = p_r0s_u / p_r0s_u.sum(-1, keepdims=True)
        return p_s_ur1 if response == 1 else p_s_ur0


if __name__ == "__main__":
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
    belief = AndBelief(num_dots, overlap_size = 4)
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
    print(EdHs.max(), EdHs.argmax(), configs[EdHs.argmax()])

    # po model
    belief = AndOrBelief(num_dots, overlap_size = 4)
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
    print(EdHs.max(), EdHs.argmax(), configs[EdHs.argmax()])

    prior = belief.prior
    for t in range(5):
        EdHs = []
        for utt in configs:
            EdH = belief.expected_info_gain(prior, utt)
            EdHs.append(EdH)
        EdHs = np.array(EdHs)
        best_idx = EdHs.argmax()
        next_utt = configs[best_idx]
        next_prior = belief.posterior(prior, next_utt, 1)
        import pdb; pdb.set_trace()
        prior = next_prior

