#!/usr/bin/env python


import models
from models import get_model_type
import data
import utils
from domain import get_domain
from corpora.reference_sentence import ReferenceSentenceCorpus


# In[4]:


# ls expts/rel3_tsel_ref_dial_model_separate/directions


# In[5]:


# ls expts/rel3_tsel_ref_dial_model_separate/directions/base_512_learned-pooling_untied_bidirectional_hidden-mention-encoder/


# In[6]:


domain = get_domain('one_common')
# unk_threshold = 10
unk_threshold = 0
model_ty = get_model_type('rnn_reference_model')

ctx_encoder_type = 'rel_attn_encoder_3'

# model_fname = 'rel2_dial_model_separate/hierarchical_only-self_mask-pad_max-pool_rel-2_feed-context-attend_no-word-attention_ablate-properties_1_best.th'


# In[7]:


import utils


# In[8]:


seed = 1
utils.set_seed(seed)
corpus = ReferenceSentenceCorpus(domain, 'data/onecommon',
                            train='train_reference_{}.txt'.format(seed),
                            valid='valid_reference_{}.txt'.format(seed),
                            test='test_reference_{}.txt'.format(seed),
                            freq_cutoff=unk_threshold, verbose=True)


# In[9]:


def load_json(file):
    import json
    with open(file, 'r') as f:
        return json.load(f)


# In[10]:


scenarios1 = load_json('data/onecommon/aaai_train_scenarios.json')
scenarios2 = load_json('data/onecommon/aaai_train_scenarios_2.json')


# In[11]:


train_scenarios = {
    scenario['uuid']: scenario
    for scenario in scenarios1
}


# In[12]:


all_scenarios = {
    scenario['uuid']: scenario
    for scenario in scenarios1 + scenarios2
}


# In[13]:


hit_count = 0
miss_count = 0
for scenario in scenarios2:
    uuid = scenario['uuid']
    if uuid in train_scenarios:
        assert train_scenarios[uuid] == scenario
        hit_count += 1
    else:
        miss_count += 1


# In[14]:


print("hits: {}".format(hit_count))
print("misses: {}".format(miss_count))


# In[15]:


batchsize=1
validset, validset_stats = corpus.valid_dataset(batchsize, shuffle=False)


# In[16]:


import torch


# In[17]:


from agent import RnnAgent


# In[18]:


from selfplay import make_parser


# In[19]:


parser = make_parser()
args = parser.parse_args([])
# args = parser.parse_args('--temperature 0.01'.split())


# In[20]:


def is_selection_function(out):
    return '<selection>' in out


# In[21]:


# display


# In[22]:


from dialog import DialogLogger


# In[23]:


from IPython.display import SVG, display, HTML


# In[24]:


def display_svgs(svgs):
    no_wrap_div = '<div style="white-space: nowrap">' + ''.join(svgs) + '</div>'
    display(HTML(no_wrap_div))


# In[25]:


SVG_SCALE=0.4


# In[26]:


def display_attn(scenario, attn, agent_id, name=None):
    attn = attn.flatten().detach().numpy()
    if name is not None:
        print("{}: {}".format(name, attn))
    display_svgs([DialogLogger._scenario_to_svg(scenario, scale=SVG_SCALE)[agent_id], DialogLogger._attention_to_svg(scenario, agent_id, attn, scale=SVG_SCALE)])


# In[27]:


def display_attns(scenario, attns, agent_id, name=None):
    svgs = [DialogLogger._scenario_to_svg(scenario, scale=SVG_SCALE)[agent_id]]
#     if name is not None:
#         print("{}: {}".format(name, attn))
    for attn in attns:
        attn = attn.flatten().detach().numpy()
        svgs.append(DialogLogger._attention_to_svg(scenario, agent_id, attn, scale=SVG_SCALE))
    if name is not None:
        print(name)
    display_svgs(svgs)


# In[28]:


# training emulation


# In[29]:


trainset, trainset_stats = corpus.train_dataset(bsz=1)


# In[30]:


# count selections


# In[31]:


from collections import Counter


# In[32]:


is_selection_counts = Counter()
for instance in validset:
    for ix, is_sel in enumerate(instance.is_selection):
        is_selection_counts[(ix, is_sel.item())] += 1


# In[33]:


weighted_sum = 0
for ix in range(11):
    true_count = is_selection_counts[(ix, True)]
    total_count = (is_selection_counts[(ix, True)] + is_selection_counts[(ix, False)])
    fraction = float(true_count) / total_count
    weighted_sum += (ix + 1) * total_count
    print(f"{ix + 1}: {true_count} / {total_count} \t= {fraction:.4f}".format(ix, fraction))
print(f"avg dialogue length: {float(weighted_sum) / sum(is_selection_counts.values()):.2f} turns")


# In[34]:


# ctx: this player's dots
# inpt / tgt: dialogue word indices (tgt are shifted by 1)
# ref_inpt


# In[35]:


# for ix in range(len(trainset)):
# #     ctx, inpts, tgts, ref_inpts, ref_tgts, sel_tgt, \
# #     [scenario_id], real_ids, agents, chat_ids, sel_idxs, \
# #     lens, rev_idxs, hid_idxs, all_num_markables = trainset[ix]
    
#     ctx, inpts, tgts, ref_inpts, ref_tgts, sel_tgt, \
#     [scenario_id], real_ids, partner_real_ids, agents, chat_ids, sel_idxs, \
#     lens, rev_idxs, hid_idxs, num_markables, \
#     is_self, partner_ref_inpts, partner_ref_tgts_our_view, all_partner_num_markables,\
#     ref_disagreements, partner_ref_disagreements = trainset[ix]

#     if 'C_d0de3d0144df405e998ae718539ff9a4' in chat_ids:
# #     if scenario_id == 'S_UoujG1GG6KscmnbN':
#         print("found it")
#         scenario = train_scenarios[scenario_id]
#         break


# In[36]:


from models.reference_predictor import ReferencePredictor


# In[37]:


from engines.beliefs import BeliefConstructor


# In[38]:


from engines.rnn_reference_engine import make_dots_mentioned_multi, make_dots_mentioned_per_ref_multi


# In[48]:


from models.reference_predictor import PragmaticReferencePredictor
from argparse import ArgumentParser


# In[50]:


def find_same_scenario(index, datasplit=validset):
    ctx, inpts, tgts, ref_inpts, ref_tgts, sel_tgt,     [scenario_id], real_ids, partner_real_ids, agents, chat_ids, sel_idxs,     lens, rev_idxs, hid_idxs, num_markables,     is_self, partner_ref_inpts, partner_ref_tgts_our_view, partner_num_markables,    ref_disagreements, partner_ref_disagreements, partner_ref_tgts, is_selections = datasplit[index]
    
    for ix, instance in enumerate(datasplit):
        if instance.scenario_ids[0] == scenario_id and ix != index:
            yield instance


# In[58]:


base = 'expts/rel3_tsel_ref_dial_model_separate/directions'
dot_recurrence = 'expts/rel3_tsel_ref_dial_model_separate/dot_recurrence'
dot_recurrence_no_beliefs = 'expts/rel3_tsel_ref_dial_model_separate/dot_recurrence_no-beliefs'
is_selection_old = 'expts/rel3_tsel_ref_dial_model_separate/is_selection_old'
is_selection_dir = 'expts/rel3_tsel_ref_dial_model_separate/is_selection'

# TODO: ep 5 -> ep 20 for with-selection
model_fnames = {
    'basic': f'{base}/base_512_learned-pooling_untied_bidirectional_hidden-mention-encoder/1_ep-30.th',
    'encoder-filtered-shared': f'{base}/base_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-shared/1_ep-30.th',
    'encoder-filtered-separate': f'{base}/base_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-separate/1_ep-30.th',
    'encoder-filtered-separate-diffs': f'{base}/base_512_extremes_learned-pooling_untied_bidirectional_hidden-mention-encoder-filtered-separate-diffs/1_ep-30.th',
#     'dot-recurrence': f'{dot_recurrence}/base_dr-self+partner-32-mention-selection/1_ep-20.th',
    'dot-recurrence-no-beliefs': f'{dot_recurrence_no_beliefs}/base_mention-selection-ref-partner/1_ep-20.th',
    'dot-recurrence-oracle-no-beliefs': f'{dot_recurrence_no_beliefs}/base_oracle_mention-selection-ref-partner/1_ep-20.th',
#     'dot-recurrence-nm-multi': f'{dot_recurrence}/base_dr-self+partner-64-mention-selection_nmpt-multi-reference_loss-expanded/1_ep-20.th',
    'dot-recurrence-nm-multi-no-beliefs': f'{dot_recurrence_no_beliefs}/base_mention-selection-ref-partner_nmpt-multi_loss-expanded/1_ep-20.th',
    'dot-recurrence-nm-multi-split-count': f'{dot_recurrence_no_beliefs}/base_split_mention-selection-ref-partner_nmpt-multi_loss-expanded_rerun_hce-cf/1_ep-20.th',
    'npm-no-lang_dri-wawm': f'{dot_recurrence_no_beliefs}/base_mention-selection-ref-partner_nmpt-multi_loss-expanded_npm-no-lang_dri-wawm/1_ep-20.th',
#     'with-selection': f'{is_selection_old}/base_hcis_1-layer_dr-in_turn/1_ep-20.th'
#     'with-selection': f'{is_selection_dir}/base_hcis_1-layer_dr-in_turn/1_ep-20.th',
    'is_selection-turn': f'{is_selection_dir}/base_turn/1_ep-7.th',
    'is_selection-turn-dr-in': f'{is_selection_dir}/base_turn/1_ep-4.th',
    'selection-no-lang': f'{dot_recurrence_no_beliefs}/base_mention-selection-ref-partner_nmpt-multi_loss-expanded_sel-no-lang_dri-wawm/1_ep-20.th',
}


# In[59]:


models = {
    name: utils.load_model(model_fname, map_location='cpu', prefix_dir=None)
    for name, model_fname in model_fnames.items()
}


# In[60]:


markable_detector = utils.load_model('serialized_models/markable_detector_with_dict_1.th', map_location='cpu', prefix_dir=None)
markable_detector.eval()
# TODO: this will no longer be necessary with markable_detector_with_dict
# markable_corpus = markable_detector.corpus_ty('one_common', 'data/onecommon',
#                                      train='train_markable_1.txt', valid='valid_markable_1.txt', test='test_markable_1.txt')
# markable_detector.word_dict = markable_corpus.word_dict


# In[61]:


models.keys()


# In[62]:


# model = models['encoder-filtered-separate-diffs']
# model = models['dot-recurrence']
# model = models['dot-recurrence-no-beliefs']
# model = models['dot-recurrence-nm-multi-no-beliefs']
# model = models['dot-recurrence-nm-multi-split-count']
# model = models['npm-no-lang_dri-wawm']
# model = models['with-selection']
model = models['selection-no-lang']


# In[63]:


reference_predictor = ReferencePredictor(model.args)


# In[64]:



# In[65]:


display_mentions = model.args.next_mention_prediction_type == 'multi_reference'


def force_contexts_and_mentions(index, datasplit=validset, candidates=1,
                                sample_temperature=0.25, argmax_temperature=0.005,
                                force_num_markables=False,
                                inference='beam', detect_markables=True,
                                min_num_mentions=0, max_num_mentions=12,
                                mention_marginals=False, mention_top_k=None,
                                ref_marginals=False, model=model,
                                for_partner=False, display_mentions=display_mentions
                               ):
    if mention_top_k is not None:
        ap = ArgumentParser()
        PragmaticReferencePredictor.add_args(ap)
        candidate_predictor = PragmaticReferencePredictor(ap.parse_args(['--l1_candidates={}'.format(mention_top_k)]))
    
    if for_partner:
        instance = next(find_same_scenario(index, datasplit=datasplit))
    else:
        instance = datasplit[index]
    
    is_selection_prediction = vars(model.args).get('is_selection_prediction')
    
    ctx, inpts, tgts, ref_inpts, ref_tgts, sel_tgt,     [scenario_id], real_ids, partner_real_ids, agents, chat_ids, sel_idxs,     lens, rev_idxs, hid_idxs, num_markables,     is_self, partner_ref_inpts, partner_ref_tgts_our_view, partner_num_markables,    ref_disagreements, partner_ref_disagreements, partner_ref_tgts, is_selections = instance
    print('scenario_id: {}'.format(scenario_id))
    
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
    agent = RnnAgent(model, args, markable_detector=markable_detector)
    agent.feed_context(ctx.flatten(),
                       belief_constructor,
                       num_markables_to_force=num_markables[0] if force_num_markables else None,
                       min_num_mentions=min_num_mentions,
                       max_num_mentions=max_num_mentions,
                      )
    agent.real_ids = real_ids
    agent.agent_id = agents[0]

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
    #         display_attn(scenario, dots_mentioned.float(), agent.agent_id, name='dots_mentioned')
            mentions = [dots_mentioned_per_ref[:,mention_ix].float()  for mention_ix in range(dots_mentioned_per_ref.size(1))]
            nm_out, _, nm_num_markables = agent.next_mention_outs[-1]
#             print('nm_num_markables: {}'.format(nm_num_markables))
#             print('num_markables: {}'.format(this_num_markables))
            if display_mentions and nm_out is not None:
#                 print(nm_out[0].size())
#                 dummy_targets = torch.zeros_like(nm_out[0].size())
                dummy_targets = torch.zeros((1, nm_num_markables[0].item(), 7)).long()
                _, nm_preds, _ = reference_predictor.forward(
                    True, dummy_targets, nm_out, nm_num_markables
                )
#                 print(nm_preds.size())
                if mention_marginals:
                    display_attns(scenario, nm_out[0].sigmoid().squeeze(1).float(), agent.agent_id, "to_mention_marginals")
                if mention_top_k is not None:
                    cands = candidate_predictor.make_candidates(nm_out, nm_num_markables)[1]
                    for ix in range(cands.size(2)):
                        display_attns(scenario, cands[:,0,ix].float(), agent.agent_id, "to_mention_candidate {}".format(ix))
                display_attns(scenario, nm_preds.squeeze(1).float(), agent.agent_id, "to_mention_pred")
            else:
                print("empty to_mention_pred")
            if display_mentions:
                display_attns(scenario, mentions, agent.agent_id, "to_mention_true")
            
#             for mention_ix in range(dots_mentioned_per_ref.size(1)):
#                 display_attn(scenario, dots_mentioned_per_ref[:,mention_ix].float(), agent.agent_id, name=f'dots_mentioned_{mention_ix}')
        else:
            dots_mentioned = torch.zeros(1, 7).bool()
            dots_mentioned_per_ref = torch.zeros(1, 0, 7).bool()
            this_num_markables = torch.LongTensor([0])
            
        if is_selection_prediction:
            print("predicted is_selection probability")
            print(agent.is_selection_outs[-1].sigmoid())
            print("is_selection: {}".format(is_selections[sentence_ix]))

        if is_self[sentence_ix]:
            if inference == 'sample':
                for sample_ix in range(candidates):
                    pred_outs, _, _, _, extra = agent.model.write(
                        agent.state, 
                        words_left, sample_temperature,
                        dots_mentioned=dots_mentioned,
                        dots_mentioned_per_ref=dots_mentioned_per_ref,
                        num_markables=this_num_markables,
                        is_selection=is_selections[sentence_ix],
                    )
                    print('sample {}\t{}'.format(sample_ix, ' '.join(agent._decode(pred_outs, agent.model.word_dict))))
            elif inference in ['beam', 'gumbel_beam']:
                _, _, decoded = agent.model.write_beam(
                    agent.state, candidates, words_left, 
                    dots_mentioned=dots_mentioned,
                    dots_mentioned_per_ref=dots_mentioned_per_ref,
                    num_markables=this_num_markables,
                    temperature=sample_temperature if inference=='gumbel_beam' else 1.0,
                    gumbel_noise=inference=='gumbel_beam',
                    is_selection=is_selections[sentence_ix],
                )
                for cand_ix, utt in enumerate(decoded):
                    print('beam {}\t{}'.format(cand_ix, ' '.join(utt)))
            else:
                raise NotImplementedError(inference)
            pred_outs, _, _, _, extra = agent.model.write(
                agent.state, 
                words_left, argmax_temperature,
                dots_mentioned=dots_mentioned,
                dots_mentioned_per_ref=dots_mentioned_per_ref,
                num_markables=this_num_markables,
                is_selection=is_selections[sentence_ix],
            )
            print('greedy\t{}'.format(' '.join(agent._decode(pred_outs, agent.model.word_dict))))
    #         print(' '.join(words_og))

            out = agent.write(
                max_words=words_left,
                force_words=[words_og],
                start_token=words_og[0],
                dots_mentioned=dots_mentioned,
                dots_mentioned_per_ref=dots_mentioned_per_ref,
                num_markables=this_num_markables,
                ref_inpt=ref_inpts[sentence_ix],
                detect_markables=detect_markables,
                is_selection=is_selections[sentence_ix],
#                 min_num_mentions=min_num_mentions,
#                 max_num_mentions=max_num_mentions,
            )
        
        else:
            out = words_og
            agent.read(words_og,
                       dots_mentioned=dots_mentioned,
                       dots_mentioned_per_ref=dots_mentioned_per_ref,
                       num_markables=this_num_markables,
                       partner_ref_inpt=partner_ref_inpts[sentence_ix],
                       partner_num_markables=partner_num_markables[sentence_ix],
                       next_num_markables_to_force=num_markables[sentence_ix + 1] if (force_num_markables and sentence_ix < len(num_markables) - 1) else None,
                       detect_markables=detect_markables,
                       min_num_mentions=min_num_mentions,
                       max_num_mentions=max_num_mentions,
                       is_selection=is_selections[sentence_ix],
                       )
        
#         agent.update_dot_h(ref_inpts[sentence_ix], partner_ref_inpts[sentence_ix],
#                             num_markables[sentence_ix], partner_num_markables[sentence_ix])
            

        print('human\t{}'.format(' '.join(out)))
    
        if is_self[sentence_ix]:
            if detect_markables:
                print('detected markables:')
                for markable in agent.markables[-1]:
                    print(markable)
                this_num_markables_pred = torch.LongTensor([len(agent.markables[-1])])
                this_ref_inpts = agent.ref_inpts[-1]
            elif ref_inpts[sentence_ix] is not None:
                this_num_markables_pred = this_num_markables
                this_ref_inpts = ref_inpt[sentence_ix]
                
                _, ref_preds, ref_stats = reference_predictor.forward(
                    ref_inpts[sentence_ix], ref_tgts[sentence_ix], agent.ref_outs[-1], this_num_markables
                )
            else:
                this_num_markables_pred = None
            if this_num_markables_pred is not None:
                dummy_targets = torch.zeros((1, this_num_markables_pred.item(), 7)).long()
                _, ref_preds, _ = reference_predictor.forward(
                    this_ref_inpts, dummy_targets, agent.ref_outs[-1], this_num_markables_pred
                )
#                 print("ref_preds size:", ref_preds.size())
                if agent.ref_outs[-1] is not None and ref_preds is not None and display_mentions:
                    if ref_marginals:
                        display_attns(scenario, agent.ref_outs[-1][0].sigmoid().squeeze(1), agent.agent_id, "ref_marginal_probs")
                    display_attns(scenario, ref_preds.squeeze(1).float(), agent.agent_id, "refs_pred")
                    display_attns(scenario, ref_tgts[sentence_ix].squeeze(0).float(), agent.agent_id, "refs_true")
#                 print(ref_stats)
            if out[0] != 'YOU':
                utt = ['YOU:'] + out
            else:
                utt = out
        else:
            if detect_markables:
                print('detected markables:')
                for markable in agent.partner_markables[-1]:
                    print(markable)
                this_partner_num_markables_pred = torch.LongTensor([len(agent.partner_markables[-1])])
                this_partner_ref_inpts = agent.partner_ref_inpts[-1]
            elif partner_ref_inpts[sentence_ix] is not None:
                this_partner_num_markables_pred = partner_num_markables[sentence_ix]
                this_partner_ref_inpts = partner_ref_inpts[sentence_ix]
            else:
                this_partner_num_markables_pred = None
            
            if this_partner_num_markables_pred is not None:
#                 _, partner_ref_preds, partner_ref_stats = reference_predictor.forward(
#                     partner_ref_inpts[sentence_ix], partner_ref_tgts_our_view[sentence_ix],
#                     agent.partner_ref_outs[-1], partner_num_markables[sentence_ix]
#                 )
                dummy_targets = torch.zeros((1, this_partner_num_markables_pred.item(), 7)).long()
                _, partner_ref_preds, partner_ref_stats = reference_predictor.forward(
                    this_partner_ref_inpts, dummy_targets,
                    agent.partner_ref_outs[-1], this_partner_num_markables_pred,
                )
    #             print(partner_ref_preds)
                if agent.partner_ref_outs[-1] is not None and partner_ref_preds is not None and display_mentions:
                    if ref_marginals:
                        display_attns(scenario, agent.partner_ref_outs[-1][0].sigmoid().squeeze(1), agent.agent_id, "partner_ref_marginal_probs")
                    display_attns(scenario, partner_ref_preds.squeeze(1).float(), agent.agent_id, "partner_refs_pred")
                    display_attns(scenario, partner_ref_tgts_our_view[sentence_ix].squeeze(0).float(), agent.agent_id, "partner_refs_true")
                    display_attns(scenario, partner_ref_tgts[sentence_ix].squeeze(0).float(), 1 - agent.agent_id, "partner_ref_true (their view)")
#                 print(partner_ref_stats)
            if out[0] == 'THEM:':
                utt = ['YOU:'] + out[1:]
            else:
                utt = ['YOU:'] + out
        
        print()
        print('-'*40)
        print()
        words_left -= len(out)
        conv.append(out)

        if is_selection_function(out):
            # sel_outs should be length 1
            selection_logits, _, _ = agent.sel_outs[-1]
            if display_mentions:
                display_attns(scenario, selection_logits.sigmoid().float(), agent.agent_id, "sel_pred")
            sel_true = torch.zeros(1,7)
            sel_true[0,sel_tgt] = 1
            if display_mentions:
                display_attns(scenario, sel_true, agent.agent_id, "sel_true")
        sentence_ix += 1
