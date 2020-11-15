
import random
import re

from cocoa.model.vocab import Vocabulary
from cocoa.core.entity import is_entity, Entity

from core.event import Event
from sessions.session import Session

import torch

class RnnSession(Session):
    def __init__(self, agent, kb, model):
        super(RnnSession, self).__init__(agent)
        self.kb = kb
        self.model = model
        self.state = {
            'selected': False,
            'quit': False,
        }

    def _is_selection(self, out):
        return len(out) == 1 and out[0] == '<selection>'

    def send(self):
        if self.state['selected']:
            return self.select()

        if self.state['quit']:
            return self.quit()

        this_partner_num_markables = torch.LongTensor([0])

        words_left = 100
        this_is_selection = False

        min_num_mentions = 1
        max_num_mentions = 3

        tokens = self.model.write(
            max_words=words_left,
            detect_markables=True,
            start_token='YOU:',
            is_selection=this_is_selection,
            inference="sample", # todo
            beam_size=1, # todo
            sample_temperature_override=1, # todo
            min_num_mentions=min_num_mentions,
            max_num_mentions=max_num_mentions,
        )

        if self._is_selection(tokens):
            return self.select()
        # TODO: nltk detok?
        # Omit the last <eos> symbol
        return self.message(' '.join(tokens[:-1]))


    def receive(self, event):
        if event.action in Event.decorative_events:
            return
        if event.action == 'select':
            self.state['selected'] = True
        elif event.action == 'quit':
            self.state['quit'] = True
        elif event.action == 'message':
            # TODO: nltk tok
            tokens = event.data.lower().strip().split() + ['<eos>']

            this_partner_num_markables = torch.LongTensor([0])
            # TODO: verify
            this_is_selection = False

            self.model.read(tokens,
                dots_mentioned=None,
                dots_mentioned_per_ref=None,
                dots_mentioned_num_markables=this_partner_num_markables,
                detect_markables=True,
                is_selection=this_is_selection,
            )

    def select(self):
        choice = self.model.choose()
        return super(RnnSession, self).select(choice)
