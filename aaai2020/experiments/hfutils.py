
from transformers import BartTokenizer

def get_bart_tokenizer():
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large")
    tokenizer.add_tokens([f"dot{i}" for i in range(8)])
    tokenizer.add_tokens(["[SEP]", "[MSEP]", "<eos>"])
    tokenizer.add_tokens(["size:", "color:", "x:", "y:", "YOU:", "THEM:"])
    #tokenizer.add_tokens(["[NONE]"])
    return tokenizer


