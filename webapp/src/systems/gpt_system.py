import argparse

import torch

from cocoa.systems.system import System
from sessions.gpt_session import GptSession
from cocoa.sessions.timed_session import TimedSessionWrapper

from annotation.transform_scenarios_to_txt import create_input

# pythonpath must have onecommon/aaai2020/experiments
import utils as utils
import pprint

import minichain
#from oc.agent.agent import Agent
from oc.agent2.agent import Agent
from engines.beliefs import BlankBeliefConstructor


CUDA = False

#torch.set_grad_enabled(False)

class Dummy:
    def __init__(self):
        self.normalize = True
        self.drop_x = False
        self.drop_y = False
        self.drop_size = False
        self.drop_color = False
        self.svg_radius = 200
        self.margin = 15
        self.base_size = 10
        self.color_range = 150
        self.size_range = 6
        self.base_color = 128
        #self.svg_grid_size = self.svg_radius * 6
        #https://github.com/dpfried/onecommon/blob/580620b9bc309625e949bb9c1dcd65063c1ba8b3/aaai2019/generate_scenarios.py

"""
class PomdpSystem(System):
    def __init__(self, name, args, timed, inference_args):
        super(PomdpSystem, self).__init__()
        self.name_ = name
        self.timed = timed
        self.inference_args = inference_args

    @classmethod
    def name(cls):
        return self.name_

    def new_session(self, agent, kb):
        model = PomdpAgent()

        # feed context, can probably save agent in init.
        ctxt = create_input(kb.items, Dummy())
        model.feed_context(ctxt)

        session = RnnSession(agent, kb, model, self.inference_args)
        if self.timed:
            session = TimedSessionWrapper(session)

        return session
"""

class GptSystem(System):
    def __init__(self, name, args, timed=False):
        super().__init__()
        self.name_ = name
        self.timed = timed
        context_manager = minichain.start_chain("tmp.txt")
        self.backend = context_manager.__enter__()
        # is it ok to never exit?

    @classmethod
    def name(cls):
        return self.name_

    def new_session(self, agent, kb):
        #model = Agent(self.backend, "codegen", "templateonly", "gpt-4")
        #model = Agent(self.backend, "shortcodegen2", "templateonly", "gpt-4")
        model = Agent(self.backend, "shortcodegen2", "templateonly", "gpt-4-0613")

        # feed context, can probably save agent in init.
        ctx = create_input(kb.items, Dummy())
        model.feed_context(ctx, flip_y=True)
        model.agent_id = agent
        model.real_ids = [int(item["id"]) for item in kb.items]

        session = GptSession(agent, kb, model)
        if self.timed:
            session = TimedSessionWrapper(session)

        return session
