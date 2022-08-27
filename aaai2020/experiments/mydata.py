
from datasets import Dataset

from domain import get_domain
from corpora.reference_sentence import ReferenceSentenceCorpus

domain = get_domain("one_common")
data = 'data/onecommon'
fold_num = 1
freq_cutoff = 20


corpus = ReferenceSentenceCorpus(
    domain, data,
    train='train_reference_{}.txt'.format(fold_num),
    valid='valid_reference_{}.txt'.format(fold_num),
    test='test_reference_{}.txt'.format(fold_num),
    freq_cutoff=freq_cutoff, verbose=True,
    max_instances_per_split=None,
    max_mentions_per_utterance=None,
    crosstalk_split=None,
    spatial_data_augmentation_on_train=False,
)

import pdb; pdb.set_trace()
