import argparse

import torch

from cocoa.systems.system import System
from sessions.rnn_session import RnnSession

from annotation.transform_scenarios_to_txt import create_input

# pythonpath must have onecommon/aaai2020/experiments
import utils as utils
import models
from agent import RnnAgent
from engines.beliefs import BlankBeliefConstructor

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

class RnnSystem(System):
    def __init__(self, name, args, model_path, markable_detector_path, timed=False):
        super(RnnSystem, self).__init__()
        self.name_ = name

        self.model = utils.load_model(model_path, prefix_dir=None)
        self.markable_detector = utils.load_model(markable_detector_path, prefix_dir=None)

        # todo: help, should probably pass in a use_cuda arg?
        if True:
            self.model.cuda()
            self.markable_detector.cuda()

    @classmethod
    def name(cls):
        return self.name_

    def new_session(self, agent, kb):
        # RnnAgent args
        parser = argparse.ArgumentParser()
        RnnAgent.add_args(parser)
        agent_args = parser.parse_args([])
        merged_args = argparse.Namespace(**utils.merge_dicts(vars(agent_args), vars(self.model.args)))
        # todo: verify
        merged_args.eps = 0

        model = RnnAgent(self.model, merged_args, markable_detector=self.markable_detector)

        # feed context, can probably save agent in init.
        ctxt = create_input(kb.items, Dummy())
        model.feed_context(ctxt, belief_constructor = BlankBeliefConstructor())

        return RnnSession(agent, kb, model)
