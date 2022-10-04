
from dataclasses import dataclass
from enum import Enum, auto

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


@dataclass
class HfDataOptions:
    properties: list[Property]

    # plan options
    unordered_rel: bool = True
    short_describe: bool = True
    plan_specific_description: bool = True

    confirmation: bool = True
    selection_leaning: bool = True
    selection: bool = True


def construct_feature_string(options):
    # construct property string
    feats = [x.name[:2] for x in options.properties]
    # add other plan options
    feats.append("ur" if options.unordered_rel else "")
    feats.append("sd" if options.short_describe else "")
    feats.append("ps" if options.plan_specific_description else "")
    # other generation options
    feats.append("c" if options.confirmation else "")
    feats.append("sl" if options.selection_leaning else "")
    feats.append("s" if options.selection else "")
    return "_".join(feats)


# hf tokenizer stuff
def get_bart_tokenizer():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    tokenizer.add_tokens([f"dot{i}" for i in range(8)])
    tokenizer.add_tokens(["[SEP]", "[MSEP]", "<eos>"])
    tokenizer.add_tokens(["size:", "color:", "x:", "y:", "YOU:", "THEM:"])
    #tokenizer.add_tokens(["[NONE]"])
    tokenizer.add_tokens(["<selection>"])
    return tokenizer

if __name__ == "__main__":
    options = HfDataOptions([
        Property.SIZE, Property.COLOR,
        Property.RX, Property.RY,
        Property.RSIZE, Property.RCOLOR,
        Property.RDIST,
    ])
    print(construct_feature_string(options))

