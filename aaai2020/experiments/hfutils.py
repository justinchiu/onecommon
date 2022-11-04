
from dataclasses import dataclass
from enum import Enum, auto
from typing import List, Tuple

from transformers import BartTokenizer


class Property(Enum):
    X = auto()
    Y = auto()
    SIZE = auto()
    COLOR = auto()

    RX = auto()
    RY = auto()
    RSIZE = auto()
    RCOLOR = auto()
    RDIST = auto()

class DescriptionFormat(Enum):
    SrcRelTgt= auto()
    SrcRelsTgt = auto()
    SrcRelTgts = auto()
    SrcsRelsTgts = auto()


@dataclass
class HfDataOptions:
    properties: list[Property]
    format: DescriptionFormat = DescriptionFormat.SrcRelTgt

    # plan options
    unordered_rel: bool = True
    short_describe: bool = True
    plan_specific_description: bool = True
    short_rel: bool = False
    config_describe: bool = False
    # OVERRIDES PLAN SPECIFIC FOR TEXT GEN
    mention_specific_description: bool = False
    
    # generation options
    confirmation: bool = True
    selection_leaning: bool = True
    selection: bool = True
    coref: bool = False

    # data options
    min_plan_size: int = 2
    max_plan_size: int = 5
    dialog_history: bool = True
    last_turn: bool = False
    last_last_turn: bool = False
    must_agree_config: bool = False
    balance: bool = False
    raw_dots: bool = False


def construct_feature_string(options):
    # construct property string
    feats = [x.name[:2] for x in options.properties]
    feats.append(options.format.name)
    # add other plan options
    feats.append("ur" if options.unordered_rel else "")
    feats.append("sd" if options.short_describe else "")
    feats.append("ps" if options.plan_specific_description else "")
    feats.append("sr" if options.short_rel else "")
    feats.append("cd" if options.config_describe else "")
    feats.append("ms" if options.mention_specific_description else "")
    # other generation options
    feats.append("c" if options.confirmation else "")
    feats.append("sl" if options.selection_leaning else "")
    feats.append("s" if options.selection else "")
    feats.append("co" if options else "")
    # data options
    feats.append(f"mps{options.min_plan_size}{options.max_plan_size}")
    feats.append("dh" if options.dialog_history else "")
    feats.append("lt" if options.last_turn else "")
    feats.append("llt" if options.last_last_turn else "")
    feats.append("ma" if options.must_agree_config else "")
    feats.append("b" if options.balance else "")
    feats.append("rd" if options.raw_dots else "")

    return "_".join(feats)


# hf tokenizer stuff
def get_bart_tokenizer():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    tokenizer.add_tokens([f"dot{i}" for i in range(8)])
    tokenizer.add_tokens(["[SEP]", "[MSEP]", "<eos>"])
    tokenizer.add_tokens(["size:", "color:", "x:", "y:", "YOU:", "THEM:"])
    #tokenizer.add_tokens(["[NONE]"])
    tokenizer.add_tokens(["<selection>", "<bom>", "<eom>", "<mention>"])
    tokenizer.add_tokens([f"<mention{i}>" for i in range(8)])
    return tokenizer

@dataclass
class GenerationExtras:
    triangle_configs: List[Tuple[int]]
    line_configs: List[Tuple[int]]


if __name__ == "__main__":
    options = HfDataOptions(
        properties = [
            Property.SIZE, Property.COLOR,
            Property.RX, Property.RY,
            Property.RSIZE, Property.RCOLOR,
            Property.RDIST,
        ],
        format = DescriptionFormat.SrcRelTgts,
        unordered_rel = True,
        short_describe = True,
        plan_specific_description = True,
        short_rel = False,
        # generation options
        confirmation = True,
        selection_leaning = True,
        selection = True,
        # data options
        max_plan_size = 5,
        dialog_history = True,
    )
    print(construct_feature_string(options))

