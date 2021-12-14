from agent import Agent, YOU_TOKEN, THEM_TOKEN



class PomdpAgent(Agent):
    def __init__(self):
        super(PomdpAgent, self).__init__()
        self.name = "Pom"
        #self.model = Pomdp()

    def feed_context(self, ctx):
        self.ctx = ctx

    def read(
        self, inpt_words, dots_mentioned=None, dots_mentioned_per_ref=None,
        dots_mentioned_num_markables=None,
        start_token=THEM_TOKEN,
        partner_ref_inpt=None, partner_num_markables=None,
        ref_tgt=None, partner_ref_tgt=None,
        detect_markables=False,
        is_selection=None,
        can_confirm=None,
    ):
        response = inpt_words[0]

        if response == "yes":
            import pdb; pdb.set_trace()
            pass
        elif response == "no":
            import pdb; pdb.set_trace()
            pass
        else:
            import pdb; pdb.set_trace()
            pass
        pass

    def write(
        self, max_words=100, force_words=None, detect_markables=True, start_token=YOU_TOKEN,
        dots_mentioned_per_ref_to_force=None,
        dots_mentioned_num_markables_to_force=None,
        ref_inpt=None,
        # used for oracle beliefs
        ref_tgt=None, partner_ref_tgt=None,
        is_selection=None,
        inference='sample',
        beam_size=1,
        sample_temperature_override=None,
        can_confirm=None,
        min_num_mentions=0,
        max_num_mentions=12,
        force_dots_mentioned=False,
    ):
        return ["hello", "<eos>"]

    def choose(self):
        import pdb; pdb.set_trace()
        pass

    def update(self, agree, reward, choice):
        import pdb; pdb.set_trace()
        pass

    def get_attention(self):
        import pdb; pdb.set_trace()
        return None



