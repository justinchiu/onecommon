
import random
import re

from cocoa.model.vocab import Vocabulary
from cocoa.core.entity import is_entity, Entity

from core.event import Event
from sessions.session import Session

import torch

from nltk import word_tokenize

class GptSession(Session):
    def __init__(self, agent, kb, model):
        super(GptSession, self).__init__(agent)
        self.kb = kb
        self.model = model
        self.state = {
            'selected': False,
            'quit': False,
        }

    def _is_selection(self, out):
        return "<selection>" in out
        #return len(out) == 1 and out[0] == '<selection>'

    def send(self):
        if self.state['selected']:
            return self.select()

        if self.state['quit']:
            return self.quit()

        words_left = 1000
        this_is_selection = None

        tokens = self.model.write()

        if self._is_selection(tokens):
            #return self.select()
            self.state["selected"] = True
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
            #tokens = event.data.lower().strip().split() + ['<eos>']
            tokens = word_tokenize(event.data.lower().strip()) + ['<eos>']
            # maybe cut out tokenization?
            self.model.read(tokens)

    def select(self):
        choice = self.model.choose()
        # convert choice to id string
        choice = self.kb.items[choice]["id"]
        return super(GptSession, self).select(choice)
