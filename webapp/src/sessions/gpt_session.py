import time
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
        #return "<selection>" in out
        return len(out) == 1 and out[0] == '<selection>'

    def send(self):
        start_time = time.time()
        if self.state['selected']:
            return self.select()

        if self.state['quit']:
            return self.quit()

        words_left = 1000
        this_is_selection = None

        tokens = self.model.write()
        print(tokens)

        if self._is_selection(tokens):
            # wait for them to select first...
            return self.select()
            #self.state["selected"] = True
        # TODO: nltk detok?
        # Omit the last <eos> symbol
        tokens = [x for x in tokens if x not in ["<eos>", "<selection>"]]
        print(f"GPT SEND TOOK {time.time() - start_time} seconds")
        return self.message(' '.join(tokens))


    def receive(self, event):
        start_time = time.time()
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
        print(f"GPT READ TOOK {time.time() - start_time} seconds")

    def select(self):
        choice = self.model.choose()
        # convert choice to id string
        choice = self.kb.items[choice]["id"]
        return super(GptSession, self).select(choice)
