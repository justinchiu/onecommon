import torch

import utils
from agent import RnnAgent
from corpora.reference_sentence import ReferenceSentenceCorpus
from domain import get_domain
from engines.beliefs import BeliefConstructor
from engines.rnn_reference_engine import make_dots_mentioned_multi, make_dots_mentioned_per_ref_multi
from models import get_model_type
from models.reference_predictor import ReferencePredictor
from selfplay import make_parser

domain = get_domain('one_common')
# unk_threshold = 10
unk_threshold = 0
model_ty = get_model_type('rnn_reference_model')

ctx_encoder_type = 'rel_attn_encoder_3'

base = 'expts/rel3_tsel_ref_dial_model_separate/directions'
dot_recurrence = 'expts/rel3_tsel_ref_dial_model_separate/dot_recurrence'

model_fnames = {
    'basic': f'{base}/base_512_learned-pooling_untied_bidirectional_hidden-mention-encoder/1_ep-30.th',
    'encoder-filtered-shared': f'{base}/base_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-shared/1_ep-30.th',
    'encoder-filtered-separate': f'{base}/base_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-separate/1_ep-30.th',
    'encoder-filtered-separate-diffs': f'{base}/base_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-separate-diffs/1_ep-30.th',
    'dot-recurrence': f'{dot_recurrence}/base_dr-self+partner-32-mention-selection/1_ep-20.th',
    'dot-recurrence-nm-multi': f'{dot_recurrence}/base_dr-self+partner-64-mention-selection_nmpt-multi-reference_expanded/1_ep-20.th',
}

seed = 1
corpus = ReferenceSentenceCorpus(domain, 'data/onecommon',
                                 train='train_reference_{}.txt'.format(seed),
                                 valid='valid_reference_{}.txt'.format(seed),
                                 test='test_reference_{}.txt'.format(seed),
                                 freq_cutoff=unk_threshold, verbose=True,
                                 max_instances_per_split=50)


def load_json(file):
    import json
    with open(file, 'r') as f:
        return json.load(f)


scenarios1 = load_json('data/onecommon/aaai_train_scenarios.json')
scenarios2 = load_json('data/onecommon/aaai_train_scenarios_2.json')

train_scenarios = {
    scenario['uuid']: scenario
    for scenario in scenarios1
}

all_scenarios = {
    scenario['uuid']: scenario
    for scenario in scenarios1 + scenarios2
}

hit_count = 0
miss_count = 0
for scenario in scenarios2:
    uuid = scenario['uuid']
    if uuid in train_scenarios:
        assert train_scenarios[uuid] == scenario
        hit_count += 1
    else:
        miss_count += 1

print("hits: {}".format(hit_count))
print("misses: {}".format(miss_count))

models = {
    name: utils.load_model(model_fname, map_location='cpu', prefix_dir=None)
    for name, model_fname in model_fnames.items()
}

batchsize = 1
validset, validset_stats = corpus.valid_dataset(batchsize)

parser = make_parser()
args = parser.parse_args('--temperature 0.25'.split())


def is_selection(out):
    return '<selection>' in out


trainset, trainset_stats = corpus.train_dataset(bsz=1)
trainset[0].scenario_ids[0] in train_scenarios

# model = models['encoder-filtered-separate-diffs']
# model = models['dot-recurrence']
model = models['dot-recurrence-nm-multi']

alice = RnnAgent(model, args)
bob = RnnAgent(model, args)

reference_predictor = ReferencePredictor(model.args)


def force_contexts_and_mentions(index, datasplit=validset, samples=0, sample_temperature=0.25,
                                argmax_temperature=0.005, force_num_markables=True):
    ctx, inpts, tgts, ref_inpts, ref_tgts, sel_tgt, [
        scenario_id], real_ids, partner_real_ids, agents, chat_ids, sel_idxs, lens, rev_idxs, hid_idxs, num_markables, is_self, partner_ref_inpts, partner_ref_tgts_our_view, partner_num_markables, ref_disagreements, partner_ref_disagreements = \
        datasplit[index]

    bsz = ctx.size(0)
    num_dots = 7

    # assume we've been training for a while, so that the beliefs are out of the warmup period (e.g. for models that switch from true->predicted)
    epoch = 1000

    dots_mentioned = make_dots_mentioned_multi(ref_tgts, model.args, bsz, num_dots)
    dots_mentioned_per_ref = make_dots_mentioned_per_ref_multi(ref_tgts, model.args, bsz, num_dots)

    partner_dots_mentioned_our_view = make_dots_mentioned_multi(
        partner_ref_tgts_our_view, model.args, bsz, num_dots
    )
    partner_dots_mentioned_our_view_per_ref = make_dots_mentioned_per_ref_multi(
        partner_ref_tgts_our_view, model.args, bsz, num_dots
    )

    belief_constructor = BeliefConstructor(
        model.args, epoch, bsz, num_dots, inpts, ref_tgts, partner_ref_tgts_our_view,
        real_ids, partner_real_ids, sel_tgt, is_self,
        partner_dots_mentioned_our_view, partner_dots_mentioned_our_view_per_ref,
        dots_mentioned, dots_mentioned_per_ref,
        ref_inpts, partner_ref_inpts,
        num_markables, partner_num_markables,
    )

    scenario = all_scenarios[scenario_id]
    agent = alice
    agent.feed_context(ctx.flatten(),
                       belief_constructor,
                       num_markables_to_force=num_markables[0] if force_num_markables else None)
    agent.real_ids = real_ids
    agent.agent_id = agents[0]

    writer = alice
    conv = []
    speaker = []
    words_left = 5000

    sentence_ix = 0

    assert len(inpts) == len(tgts)

    while sentence_ix < len(inpts):
        words_og = [corpus.word_dict.idx2word[ix] for ix in inpts[sentence_ix].flatten().numpy()]

        if ref_inpts[sentence_ix] is not None:
            dots_mentioned = (ref_tgts[sentence_ix].sum(dim=1) > 0)
            dots_mentioned_per_ref = ref_tgts[sentence_ix]
            this_num_markables = num_markables[sentence_ix]
            #         display_attn(scenario, dots_mentioned.float(), writer.agent_id, name='dots_mentioned')
            mentions = [dots_mentioned_per_ref[:, mention_ix].float() for mention_ix in
                        range(dots_mentioned_per_ref.size(1))]
            nm_out, _, nm_num_markables = writer.next_mention_outs[-1]
            #             print('nm_num_markables: {}'.format(nm_num_markables))
            #             print('num_markables: {}'.format(this_num_markables))
            if nm_out is not None:
                #                 print(nm_out[0].size())
                _, nm_preds, _ = reference_predictor.forward(
                    True, dots_mentioned_per_ref, nm_out, nm_num_markables
                )
                #                 print(nm_preds.size())
        else:
            dots_mentioned = torch.zeros(1, 7).bool()
            dots_mentioned_per_ref = torch.zeros(1, 0, 7).bool()
            this_num_markables = torch.LongTensor([0])

        if is_self[sentence_ix]:
            for sample_ix in range(samples):
                pred_outs, _, _, _, extra = writer.model.write(
                    writer.state,
                    words_left, sample_temperature,
                    dots_mentioned=dots_mentioned,
                    dots_mentioned_per_ref=dots_mentioned_per_ref,
                    num_markables=this_num_markables
                )
                print('sample {}\t{}'.format(sample_ix, ' '.join(writer._decode(pred_outs, writer.model.word_dict))))
            pred_outs, _, _, _, extra = writer.model.write_beam(
                writer.state,
                10,
                words_left,
                dots_mentioned=dots_mentioned,
                dots_mentioned_per_ref=dots_mentioned_per_ref,
                num_markables=this_num_markables
            )
            print('argmax\t{}'.format(' '.join(writer._decode(pred_outs, writer.model.word_dict))))
            #         print(' '.join(words_og))

            out = writer.write(
                max_words=words_left,
                force_words=[words_og],
                start_token=words_og[0],
                dots_mentioned=dots_mentioned,
                dots_mentioned_per_ref=dots_mentioned_per_ref,
                num_markables=this_num_markables,
                ref_inpt=ref_inpts[sentence_ix],
            )

        else:
            out = words_og
            writer.read(words_og,
                        dots_mentioned=dots_mentioned,
                        dots_mentioned_per_ref=dots_mentioned_per_ref,
                        num_markables=this_num_markables,
                        partner_ref_inpt=partner_ref_inpts[sentence_ix],
                        partner_num_markables=partner_num_markables[sentence_ix],
                        next_num_markables_to_force=num_markables[sentence_ix + 1] if (
                            force_num_markables and sentence_ix < len(num_markables) - 1) else None,
                        )

        print('human\t{}'.format(' '.join(out)))

        if is_self[sentence_ix]:
            if ref_inpts[sentence_ix] is not None:
                _, ref_preds, ref_stats = reference_predictor.forward(
                    ref_inpts[sentence_ix], ref_tgts[sentence_ix], writer.ref_outs[-1], this_num_markables
                )
        else:
            if partner_ref_inpts[sentence_ix] is not None:
                _, partner_ref_preds, partner_ref_stats = reference_predictor.forward(
                    partner_ref_inpts[sentence_ix], partner_ref_tgts_our_view[sentence_ix],
                    writer.partner_ref_outs[-1], partner_num_markables[sentence_ix]
                )

        print()
        print('-' * 40)
        print()
        words_left -= len(out)
        conv.append(out)
        if is_selection(out) or words_left <= 1:
            break
        sentence_ix += 1

force_contexts_and_mentions(1)
