import copy
from collections import defaultdict, namedtuple
from typing import Union

import torch.nn.init
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence, pad_sequence
import einops

import corpora.reference
import corpora.reference_sentence
import models
from corpora import data
from domain import get_domain
from engines import Criterion
from engines.beliefs import BeliefConstructor
from engines.rnn_reference_engine import RnnReferenceEngine, HierarchicalRnnReferenceEngine, CAN_CONFIRM_VALUES
from models import reference_predictor
from models.attention_layers import FeedForward, AttentionLayer, StructuredAttentionLayer, \
    StructuredTemporalAttentionLayer
from models.ctx_encoder import *
from models.reference_predictor import PragmaticReferencePredictor, ReferencePredictor, make_candidates
from utils import set_temporary_default_tensor_type
from models.utils import lengths_to_mask

from torch.distributions import Gumbel

BIG_NEG = -1e6

MAX_NUM_MENTIONS = 12

_State = namedtuple('_State', [
    'args', 'bsz', 'ctx', 'ctx_h', 'ctx_differences', 'reader_and_writer_lang_h',
    'dot_h_maybe_multi', 'dot_h_maybe_multi_structured', 'belief_constructor',
    'turn',
])

NextMentionRollouts = namedtuple('NextMentionRollouts', [
    'current_sel_probs', 'num_markables_per_candidate', 'candidate_indices', 'candidate_dots', 'candidate_nm_scores', 'rollout_sel_probs',
])

NextMentionLatents = namedtuple('NextMentionLatents', [
    'latent_states', 'dots_mentioned_num_markables', 'stop_losses', 'ctx_h_with_beliefs'
])

class State(_State):

    def dot_h(self):
        if vars(self.args).get('dot_recurrence_split', False):
            assert isinstance(self.dot_h_maybe_multi, tuple)
            dot_h = torch.cat(self.dot_h_maybe_multi, -1)
        else:
            dot_h = self.dot_h_maybe_multi
        return dot_h

    def _maybe_add_dot_h_to_beliefs(self, beliefs, name):
        if self.args.dot_recurrence and name in self.args.dot_recurrence_in:
            dot_h = self.dot_h()
            if beliefs is None:
                return dot_h
            else:
                return torch.cat((beliefs, dot_h), -1)
        else:
            return beliefs

    def make_beliefs(self, name, timestep, partner_ref_outs, ref_outs):
        if self.belief_constructor is not None:
            beliefs = self.belief_constructor.make_beliefs(name, timestep, partner_ref_outs, ref_outs)
        else:
            beliefs = None
        add_name = name
        # TODO: fix naming mismatch
        if add_name == 'mention':
            add_name = 'next_mention'
        beliefs = self._maybe_add_dot_h_to_beliefs(beliefs, add_name)
        return beliefs

    def expand_bsz(self, new_bsz):
        assert self.bsz == 1
        ctx = self.ctx.expand(new_bsz, -1)
        ctx_h = self.ctx_h.expand(new_bsz, -1, -1)
        ctx_differences = self.ctx_differences.expand(new_bsz, -1, -1)
        _reader_lang_h, _writer_lang_h = self.reader_and_writer_lang_h
        reader_lang_h = _reader_lang_h.expand(-1, new_bsz, -1)
        writer_lang_h = _writer_lang_h.expand(-1, new_bsz, -1)
        dot_h_maybe_multi = self.dot_h_maybe_multi.expand(new_bsz, -1, -1)
        if self.dot_h_maybe_multi_structured is not None:
            raise NotImplementedError()
        dot_h_maybe_multi_structured = None
        return self._replace(
            bsz=new_bsz, ctx=ctx, ctx_h=ctx_h, ctx_differences=ctx_differences,
            reader_and_writer_lang_h=(reader_lang_h, writer_lang_h), dot_h_maybe_multi=dot_h_maybe_multi,
            dot_h_maybe_multi_structured=dot_h_maybe_multi_structured,
        )

    def mask(self, mask_to_keep):
        sub_bsz = mask_to_keep.size()
        mask_to_keep = mask_to_keep.bool()
        ctx = self.ctx[mask_to_keep]
        ctx_h = self.ctx_h[mask_to_keep]
        ctx_differences = self.ctx_differences[mask_to_keep]
        _reader_lang_h, _writer_lang_h = self.reader_and_writer_lang_h
        reader_lang_h = _reader_lang_h[:,mask_to_keep]
        writer_lang_h = _writer_lang_h[:,mask_to_keep]
        dot_h_maybe_multi = self.dot_h_maybe_multi[mask_to_keep] if self.dot_h_maybe_multi is not None else None
        if self.dot_h_maybe_multi_structured is not None:
            raise NotImplementedError()
        dot_h_maybe_multi_structured = None
        return self._replace(
            bsz=sub_bsz, ctx=ctx, ctx_h=ctx_h, ctx_differences=ctx_differences,
            reader_and_writer_lang_h=(reader_lang_h, writer_lang_h), dot_h_maybe_multi=dot_h_maybe_multi,
            dot_h_maybe_multi_structured=dot_h_maybe_multi_structured,
        )


class RnnReferenceModel(nn.Module):
    corpus_ty = corpora.reference.ReferenceCorpus
    engine_ty = RnnReferenceEngine

    @classmethod
    def add_args(cls, parser):

        parser.add_argument('--nembed_word', type=int, default=128,
                            help='size of word embeddings')
        parser.add_argument('--nhid_rel', type=int, default=64,
                            help='size of the hidden state for the language module')
        parser.add_argument('--nembed_ctx', type=int, default=128,
                            help='size of context embeddings')
        parser.add_argument('--nembed_cond', type=int, default=128,
                            help='size of condition embeddings')
        parser.add_argument('--nhid_lang', type=int, default=128,
                            help='size of the hidden state for the language module')
        parser.add_argument('--nhid_strat', type=int, default=128,
                            help='size of the hidden state for the strategy module')
        parser.add_argument('--nhid_attn', type=int, default=64,
                            help='size of the hidden state for the attention module')
        parser.add_argument('--nhid_sel', type=int, default=64,
                            help='size of the hidden state for the selection module')
        parser.add_argument('--share_attn', action='store_true', default=False,
                            help='share attention modules for selection and language output')
        parser.add_argument('--separate_attn', action='store_true', default=False,
                            help="don't share the first layer of the attention module")
        parser.add_argument('--tie_reference_attn', action='store_true')

        parser.add_argument('--hid2output',
                            choices=['activation-final', '1-hidden-layer', '2-hidden-layer'],
                            default='activation-final')

        parser.add_argument('--selection_attention', action='store_true')
        parser.add_argument('--feed_context', action='store_true')
        parser.add_argument('--feed_context_attend', action='store_true')
        parser.add_argument('--feed_context_attend_separate', action='store_true')

        parser.add_argument('--hidden_context', action='store_true')
        parser.add_argument('--hidden_context_is_selection', action='store_true')
        parser.add_argument('--hidden_context_confirmations', action='store_true')
        parser.add_argument('--hidden_context_confirmations_in',
                            choices=['generation', 'next_mention'], nargs='*', default=['generation'])
        parser.add_argument('--confirmations_resolution_strategy', choices=['any', 'all', 'half'], default='any')
        parser.add_argument('--hidden_context_mention_encoder', action='store_true')
        parser.add_argument('--hidden_context_mention_encoder_bidirectional', action='store_true')
        parser.add_argument('--hidden_context_mention_encoder_type', choices=[
            'full','filtered-separate','filtered-shared'
        ], default='full')
        parser.add_argument('--hidden_context_mention_encoder_diffs', action='store_true')
        parser.add_argument('--hidden_context_mention_encoder_property_diffs', action='store_true')
        parser.add_argument('--hidden_context_mention_encoder_dot_recurrence', action='store_true')
        parser.add_argument('--hidden_context_mention_encoder_count_features', action='store_true')
        parser.add_argument('--hidden_context_mention_encoder_attention', action='store_true')

        parser.add_argument('--untie_grus',
                            action='store_true',
                            help="don't use the same weights for the reader and writer")
        parser.add_argument('--bidirectional_reader',
                            action='store_true',
                            help="use a bidirectional reader")

        parser.add_argument('--attention_type', choices=['softmax', 'sigmoid'], default='softmax')

        parser.add_argument('--learned_pooling', action='store_true')

        # beliefs / dot-conditioning
        parser.add_argument('--no_word_attention',
                            action='store_true',
                            help="don't attend to the context in the word output layer")

        parser.add_argument('--word_attention_constrained',
                            action='store_true',
                            help="don't allow attention to dots that aren't mentioned in the utterance")

        parser.add_argument('--feed_attention_constrained',
                            action='store_true',
                            help="don't allow attention to dots that aren't mentioned in the utterance")

        parser.add_argument('--hidden_attention_constrained',
                            action='store_true',
                            help="don't allow attention to dots that aren't mentioned in the utterance")

        parser.add_argument('--mark_dots_mentioned',
                            action='store_true',
                            help='give an indicator feature for whether the given dot should be mentioned')

        parser.add_argument('--only_first_mention',
                            action='store_true',
                            help='when marking dots mentioned (either in mark_dots_mentioned, or in one of the oracle selection beliefs), only mark the first mention in the utterance')

        parser.add_argument('--marks_in_word_prediction', action='store_true',
                            help='in addition to marking context in attention, mark context in the word prediction layer')

        parser.add_argument('--detach_beliefs', action='store_true',
                            help='don\'t backprop through the belief prediction network')

        # dot recurrence beliefs
        parser.add_argument('--dot_recurrence', nargs='+', choices=['self', 'partner'])
        parser.add_argument('--dot_recurrence_dim', type=int, default=32)
        parser.add_argument('--dot_recurrence_oracle', action='store_true')
        parser.add_argument('--dot_recurrence_oracle_for', nargs='+', choices=['self', 'partner'], default=['self', 'partner'])
        parser.add_argument('--dot_recurrence_mention_attention', action='store_true')
        parser.add_argument('--dot_recurrence_structured',
                            action='store_true', help='use one hidden state for each of the 2^7 configurations')
        parser.add_argument('--dot_recurrence_structured_layers', type=int, default=0)
        parser.add_argument('--dot_recurrence_uniform',
                            action='store_true',
                            help='all dots get the same update from language features (i.e. uniform attention weights) [seems to hurt performance]')
        parser.add_argument('--dot_recurrence_in',
                            nargs='+',
                            choices=[
                                'selection', 'next_mention', 'generation', 'ref', 'partner_ref', 'is_selection',
                                'next_mention_latents'
                            ],
                            default=[])
        parser.add_argument('--dot_recurrence_split', action='store_true')
        parser.add_argument('--dot_recurrence_inputs',
                            nargs='+',
                            choices=[
                                'weighted_hidden', 'weights_average', 'weights_max',
                                'predicted_hidden', 'predicted_average', 'predicted_max',
                            ],
                            default='weighted_hidden')

        # auxiliary models
        parser.add_argument('--partner_reference_prediction', action='store_true')

        parser.add_argument('--next_mention_prediction', action='store_true')
        parser.add_argument('--next_mention_prediction_type', choices=['collapsed', 'multi_reference'], default='collapsed')
        parser.add_argument('--next_mention_length_prediction_type',
                            choices=['bernoulli', 'categorical'],
                            default='bernoulli')
        parser.add_argument('--next_mention_prediction_no_lang', action='store_true')
        parser.add_argument('--next_mention_prediction_latent_layers', type=int, default=0)

        parser.add_argument('--selection_prediction_no_lang', action='store_true')

        parser.add_argument('--is_selection_prediction', action='store_true')
        parser.add_argument('--is_selection_prediction_layers', type=int, default=0)
        # parser.add_argument('--is_selection_prediction_turn_feature', action='store_true')
        # parser.add_argument('--is_selection_prediction_dot_context', action='store_true')
        parser.add_argument('--is_selection_prediction_features', nargs='*',
                            choices=['language_state', 'dot_context', 'turn'], default=[])

        AttentionLayer.add_args(parser)
        StructuredAttentionLayer.add_args(parser)
        StructuredTemporalAttentionLayer.add_args(parser)

        BeliefConstructor.add_belief_args(parser)

    def __init__(self, word_dict, args):
        super(RnnReferenceModel, self).__init__()

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.args = args
        self.num_ent = domain.num_ent()

        # define modules:
        self.word_embed = nn.Embedding(len(self.word_dict), args.nembed_word)

        self.spatial_replacements = {
            self.word_dict.word2idx[k]: self.word_dict.word2idx[v]
            for k, v in corpora.reference.ReferenceCorpus.SPATIAL_REPLACEMENTS.items()
            if k in self.word_dict.word2idx and v in self.word_dict.word2idx
        }

        ctx_encoder_ty = models.get_ctx_encoder_type(args.ctx_encoder_type)
        self.ctx_encoder = ctx_encoder_ty(domain, args)

        if self.args.feed_context_attend_separate:
            assert self.args.feed_context_attend

        if self.args.feed_context_attend:
            assert self.args.feed_context

        if self.args.feed_context:
            self.feed_ctx_layer = nn.Sequential(
                torch.nn.Linear(args.nembed_ctx, args.nembed_ctx),
                # added this
                torch.nn.ReLU(),
            )
            gru_input_size = args.nembed_word + args.nembed_ctx
        else:
            gru_input_size = args.nembed_word

        if self.args.hidden_context:
            assert self.args.mark_dots_mentioned
            self.hidden_ctx_layer = nn.Linear(args.nembed_ctx, args.nembed_ctx)
            self.hidden_ctx_gate = nn.Sequential(
                torch.nn.Linear(args.nembed_ctx+args.nhid_lang, args.nhid_lang),
                nn.Sigmoid()
            )
            self.hidden_ctx_addition = nn.Sequential(
                torch.nn.Linear(args.nembed_ctx, args.nhid_lang),
                nn.Tanh(),
            )
            if self.args.hidden_context_mention_encoder:
                hce_input_dim = args.nembed_ctx
                if self.args.hidden_context_mention_encoder_diffs:
                    hce_input_dim += args.nembed_ctx
                    diff_input_dim = args.nembed_ctx
                    if self.args.hidden_context_mention_encoder_property_diffs:
                        hce_input_dim += 4
                        diff_input_dim += 4
                    self.hidden_ctx_encoder_diff_none = nn.Parameter(
                        torch.zeros(diff_input_dim), requires_grad=True,
                    )
                    self.hidden_ctx_encoder_diff_pad = nn.Parameter(
                        torch.zeros(diff_input_dim), requires_grad=True,
                    )
                if self.args.hidden_context_mention_encoder_dot_recurrence:
                    dr_dim = args.dot_recurrence_dim * (2 if self.args.dot_recurrence_split else 1)
                else:
                    dr_dim = 0
                hce_input_dim += dr_dim
                if self.args.hidden_context_mention_encoder_count_features:
                    hce_input_dim += 1

                hce_encoder_output_dim = args.nembed_ctx
                if self.args.hidden_context_mention_encoder_bidirectional:
                    assert hce_encoder_output_dim % 2 == 0
                    hce_encoder_output_dim //= 2
                self.hidden_ctx_encoder = nn.LSTM(
                    hce_input_dim, hce_encoder_output_dim, 1, batch_first=True,
                    bidirectional=self.args.hidden_context_mention_encoder_bidirectional,
                )
                self.hidden_ctx_encoder_no_markables = nn.Parameter(
                    torch.zeros(args.nembed_ctx), requires_grad=True
                )
                self.hidden_ctx_encoder_no_dots = nn.Parameter(
                    torch.zeros(args.nembed_ctx + dr_dim), requires_grad=True
                )

                if self.args.hidden_context_mention_encoder_property_diffs:
                    self.hidden_ctx_encoder_no_dots_prop_diff = nn.Parameter(
                        torch.zeros(4), requires_grad=True
                    )
                if self.args.hidden_context_mention_encoder_type == 'filtered-separate':
                    ctx_encoder_ty = models.get_ctx_encoder_type(args.ctx_encoder_type)
                    self.hidden_ctx_encoder_relational = ctx_encoder_ty(domain, args)

                if self.args.hidden_context_mention_encoder_attention:
                    self.mention_encoder_attention = nn.Bilinear(args.nembed_ctx, args.nhid_lang, 1)
                    self.mention_encoder_attention_no_mentions_emb = nn.Parameter(
                        torch.zeros(args.nembed_ctx), requires_grad=True
                    )

            if self.args.hidden_context_is_selection:
                self.hidden_ctx_is_selection_embeddings = nn.Embedding(2, args.nhid_lang)
                self.hidden_ctx_is_selection_weight = nn.Parameter(torch.zeros((1,)))

            if self.args.hidden_context_confirmations:
                # have / don't have / other person's turn
                if 'generation' in self.args.hidden_context_confirmations_in:
                    self.hidden_ctx_confirmations_embeddings = nn.Embedding(3, args.nhid_lang, padding_idx=2)
                    self.hidden_ctx_confirmations_weight = nn.Parameter(torch.zeros((1,)))
                if 'next_mention' in self.args.hidden_context_confirmations_in:
                    self.hidden_ctx_confirmations_nm_embeddings = nn.Embedding(3, args.nhid_lang, padding_idx=2)
                    self.hidden_ctx_confirmations_nm_weight = nn.Parameter(torch.zeros((1,)))

        self.reader = nn.GRU(
            input_size=gru_input_size,
            hidden_size=args.nhid_lang,
            bias=True, bidirectional=args.bidirectional_reader)

        self.writer = nn.GRU(
            input_size=gru_input_size,
            hidden_size=args.nhid_lang,
            bias=True)

        self.writer_cell = nn.GRUCell(
            input_size=gru_input_size,
            hidden_size=args.nhid_lang,
            bias=True)

        if self.args.dot_recurrence:

            self.dot_rec_cell = nn.GRUCell(
                input_size=args.nhid_lang,
                hidden_size=args.dot_recurrence_dim,
                bias=True)
            self.dot_rec_init = nn.Parameter(torch.zeros(args.dot_recurrence_dim))
            if vars(self.args).get('dot_recurrence_structured', False):
                self.dot_rec_structured_init = nn.Parameter(torch.zeros(args.dot_recurrence_dim))
            if self.args.dot_recurrence_split:
                self.dot_rec_partner_cell = nn.GRUCell(
                    input_size=args.nhid_lang,
                    hidden_size=args.dot_recurrence_dim,
                    bias=True)
                self.dot_rec_partner_init = nn.Parameter(torch.zeros(args.dot_recurrence_dim))
                if vars(self.args).get('dot_recurrence_structured', False):
                    self.dot_rec_partner_structured_init = nn.Parameter(torch.zeros(args.dot_recurrence_dim))
                self.dot_recurrence_embeddings = 2
            else:
                self.dot_recurrence_embeddings = 1

            self.dot_rec_ref_pooling_weights = nn.Parameter(torch.zeros(3), requires_grad=True)
            self.dot_rec_partner_ref_pooling_weights = copy.deepcopy(self.dot_rec_ref_pooling_weights)

            ref_transform_input_dim = 0
            if 'weighted_hidden' in self.args.dot_recurrence_inputs:
                ref_transform_input_dim += args.nhid_lang * self.num_reader_directions
            if 'weights_average' in self.args.dot_recurrence_inputs:
                ref_transform_input_dim += 1
            if 'weights_max' in self.args.dot_recurrence_inputs:
                ref_transform_input_dim += 1
            if 'predicted_hidden' in self.args.dot_recurrence_inputs:
                ref_transform_input_dim += args.nhid_lang * self.num_reader_directions
            if 'predicted_average' in self.args.dot_recurrence_inputs:
                ref_transform_input_dim += 1
            if 'predicted_max' in self.args.dot_recurrence_inputs:
                ref_transform_input_dim += 1

            self.dot_rec_ref_transform = nn.Linear(ref_transform_input_dim, args.nhid_lang)
            self.dot_rec_partner_ref_transform = copy.deepcopy(self.dot_rec_ref_transform)

            if args.dot_recurrence_mention_attention:
                self.dot_recurrence_mention_attention = nn.Linear(ref_transform_input_dim, 1)

            self.dot_recurrence_weight_temperature = 1.0

        # nhid_lang
        # TODO: pass these
        self.reader_init_h = torch.nn.Parameter(torch.zeros(args.nhid_lang), requires_grad=True)
        if args.bidirectional_reader:
            self.reader_init_h_reverse = torch.nn.Parameter(torch.zeros(args.nhid_lang), requires_grad=True)

        # tie the weights between reader and writer?
        if not args.untie_grus:
            self.writer.weight_ih_l0 = self.reader.weight_ih_l0
            self.writer.weight_hh_l0 = self.reader.weight_hh_l0
            self.writer.bias_ih_l0 = self.reader.bias_ih_l0
            self.writer.bias_hh_l0 = self.reader.bias_hh_l0
            self.writer_init_h = self.reader_init_h
        else:
            # n_layers * n_directions x nhid_lang
            self.writer_init_h = torch.nn.Parameter(torch.zeros(args.nhid_lang), requires_grad=True)

        self.writer_cell.weight_ih = self.writer.weight_ih_l0
        self.writer_cell.weight_hh = self.writer.weight_hh_l0
        self.writer_cell.bias_ih = self.writer.bias_ih_l0
        self.writer_cell.bias_hh = self.writer.bias_hh_l0

        # if args.attentive_selection_encoder:
        #     self.selection_attention = nn.Sequential(
        #
        #     )

        if args.no_word_attention:
            h2o_input_dim = args.nhid_lang
        else:
            h2o_input_dim = args.nhid_lang + args.nembed_ctx
            if vars(self.args).get('marks_in_word_prediction', False):
                if args.mark_dots_mentioned:
                    h2o_input_dim += 1
                if len(args.generation_beliefs) > 0:
                    h2o_input_dim += len(args.generation_beliefs)
            if self.args.hidden_context_mention_encoder_attention:
                h2o_input_dim += args.nembed_ctx

        if args.hid2output == 'activation-final':
            self.hid2output = nn.Sequential(
                nn.Linear(h2o_input_dim, args.nembed_word),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                )
        elif args.hid2output == '1-hidden-layer':
            self.hid2output = FeedForward(1, h2o_input_dim, args.nhid_lang, args.nembed_word, dropout_p=args.dropout)
        elif args.hid2output == '2-hidden-layer':
            self.hid2output = FeedForward(2, h2o_input_dim, args.nhid_lang, args.nembed_word, dropout_p=args.dropout)
        else:
            raise ValueError("invalid --hid2output {}".format(args.hid2output))

        ref_attn_names = ['ref']
        if args.partner_reference_prediction:
            ref_attn_names.append('ref_partner')

        if args.next_mention_prediction:
            ref_attn_names.append('next_mention')
            if self.args.next_mention_prediction_no_lang:
                self.next_mention_start_emb = nn.Parameter(torch.zeros(args.nhid_lang), requires_grad=True)

        if self.args.selection_prediction_no_lang:
            self.selection_start_emb = nn.Parameter(torch.zeros(args.nhid_lang * 2), requires_grad=True)

        lang_attn_names = ['lang', 'feed']

        if args.hidden_context:
            lang_attn_names.append('hidden')

        if args.mention_beliefs:
            assert args.next_mention_prediction

        if args.next_mention_prediction_type == 'multi_reference':
            self.next_mention_cell = nn.GRUCell(
                args.nhid_lang, args.nhid_lang, bias=True
            )
            transform_input_dim = args.nhid_lang
            if 'next_mention_latents' in args.dot_recurrence_in:
                transform_input_dim += args.dot_recurrence_dim * (2 if self.args.dot_recurrence_split else 1)

            def _make_layer(input_dim, output_dim):
                if args.next_mention_prediction_latent_layers == 0:
                    return nn.Linear(input_dim, output_dim)
                else:
                    return FeedForward(
                        args.next_mention_prediction_latent_layers, input_dim, 128, output_dim,
                        dropout_p=args.dropout
                    )

            self.next_mention_transform = _make_layer(transform_input_dim, args.nhid_lang)
            if self.args.next_mention_length_prediction_type == 'bernoulli':
                self.next_mention_stop = _make_layer(args.nhid_lang, 1)
            else:
                self.next_mention_length = _make_layer(transform_input_dim, MAX_NUM_MENTIONS+1)
                self.next_mention_length_crit = nn.CrossEntropyLoss(reduction='none')

        if args.structured_attention or args.structured_temporal_attention:
            assert not args.share_attn
            attention_constructors = {
                'ref': StructuredTemporalAttentionLayer if args.structured_temporal_attention else StructuredAttentionLayer,
                'ref_partner': StructuredTemporalAttentionLayer if args.structured_temporal_attention else StructuredAttentionLayer,
                'next_mention': StructuredTemporalAttentionLayer \
                    if args.structured_temporal_attention and args.next_mention_prediction_type == 'multi_reference' \
                    else StructuredAttentionLayer,
                'sel': AttentionLayer,
                'lang': AttentionLayer, # todo: consider structured attention with sigmoids here
                'feed': AttentionLayer,
                'hidden': AttentionLayer,
            }
        else:
            attention_constructors = defaultdict(lambda: AttentionLayer)

        if args.share_attn:
            assert not args.separate_attn
            # TODO: get rid of layers here
            assert not args.mark_dots_mentioned
            assert not args.selection_beliefs
            assert not args.generation_beliefs
            assert not args.mention_beliefs
            assert not self.args.bidirectional_reader
            self.attn = AttentionLayer(args, 2, args.nhid_lang + args.nembed_ctx, args.nhid_attn, dropout_p=args.dropout, language_dim=args.nhid_lang)
            if self.args.feed_context_attend_separate:
                self.feed_attn = AttentionLayer(args, 2, args.nhid_lang + args.nembed_ctx, args.nhid_attn, dropout_p=args.dropout, language_dim=args.nhid_lang)
            else:
                self.feed_attn = None
        elif args.separate_attn:
            for attn_name in lang_attn_names + ['sel'] + ref_attn_names:
                if attn_name in lang_attn_names or attn_name == 'next_mention':
                    # writer
                    lang_input_dim = self.args.nhid_lang
                else:
                    lang_input_dim = self.args.nhid_lang * self.num_reader_directions
                input_dim = lang_input_dim + args.nembed_ctx
                if args.mark_dots_mentioned and attn_name in lang_attn_names:
                    # TODO: consider tiling the indicator feature across more dimensions
                    input_dim += 1
                if attn_name == 'sel':
                    if args.selection_beliefs:
                        input_dim += len(args.selection_beliefs)
                    if args.dot_recurrence and 'selection' in args.dot_recurrence_in:
                        input_dim += args.dot_recurrence_dim * self.dot_recurrence_embeddings
                if attn_name in lang_attn_names:
                    if args.generation_beliefs:
                        input_dim += len(args.generation_beliefs)
                    if args.dot_recurrence and 'generation' in args.dot_recurrence_in:
                        input_dim += args.dot_recurrence_dim * self.dot_recurrence_embeddings
                if attn_name == 'next_mention':
                    if args.mention_beliefs:
                        input_dim += len(args.mention_beliefs)
                    if args.dot_recurrence and 'next_mention' in args.dot_recurrence_in:
                        input_dim += args.dot_recurrence_dim * self.dot_recurrence_embeddings
                if attn_name == 'ref':
                    if args.ref_beliefs:
                        input_dim += len(args.ref_beliefs)
                    if args.dot_recurrence and 'ref' in args.dot_recurrence_in:
                        input_dim += args.dot_recurrence_dim * self.dot_recurrence_embeddings
                if attn_name == 'ref_partner':
                    if args.partner_ref_beliefs:
                        input_dim += len(args.partner_ref_beliefs)
                    # TODO: fix ref_partner vs partner_ref naming
                    if args.dot_recurrence and 'partner_ref' in args.dot_recurrence_in:
                        input_dim += args.dot_recurrence_dim * self.dot_recurrence_embeddings
                if self.args.tie_reference_attn and attn_name == 'ref_partner':
                    pass
                else:
                    setattr(
                        self,
                        self._attention_name(attn_name),
                        attention_constructors[attn_name](args, 2, input_dim, args.nhid_attn, dropout_p=args.dropout, language_dim=lang_input_dim)
                    )
            if self.args.tie_reference_attn and self.args.partner_reference_prediction:
                self.ref_partner_attn = self.ref_attn
        else:
            assert not args.mark_dots_mentioned
            assert not args.selection_beliefs
            assert not args.generation_beliefs

            self.attn_prefix = nn.Sequential(
                nn.Linear(args.nhid_lang + args.nembed_ctx, args.nhid_sel),
                nn.ReLU(),
                nn.Dropout(args.dropout))
            for attn_name in ['lang', 'sel'] + ref_attn_names:
                setattr(
                    self,
                    self._attention_name(attn_name),
                    attention_constructors[attn_name](args, 1, args.nhid_sel, args.nhid_sel, dropout_p=args.dropout, language_dim=lang_input_dim)
                )
            if self.args.feed_context_attend_separate:
                # TODO: why separate hidden dim for this?
                self.feed_attn = attention_constructors['feed'](args, 1, args.nhid_sel, args.nhid_attn, dropout_p=args.dropout, language_dim=lang_input_dim)
            else:
                self.feed_attn = None

        if self.args.selection_attention:
            self.ctx_layer = nn.Sequential(
                torch.nn.Linear(args.nembed_ctx, args.nhid_attn),
                # nn.ReLU(),
                # nn.Dropout(args.dropout),
                # torch.nn.Linear(args.nhid_attn, args.nhid_attn),
            )

            self.selection_attention_layer = nn.Sequential(
                # takes nembed_ctx and the output of ctx_layer
                torch.nn.Linear(args.nembed_ctx + args.nhid_attn, args.nhid_attn),
                nn.ReLU(),
                nn.Dropout(args.dropout),
                torch.nn.Linear(args.nhid_attn, 1)
            )

        self.dropout = nn.Dropout(args.dropout)

        # learned pooling
        if self.args.learned_pooling:
            self.ref_pooling_weights = nn.Parameter(torch.zeros(3), requires_grad=True)
            self.partner_ref_pooling_weights = nn.Parameter(torch.zeros(3), requires_grad=True)

        # mask for disabling special tokens when generating sentences
        self.special_token_mask = make_mask(len(word_dict),
            [word_dict.get_idx(w) for w in ['<unk>', 'YOU:', 'THEM:', '<pad>']])

        # init
        self.word_embed.weight.data.uniform_(-args.init_range, args.init_range)
        # init_rnn(self.reader, args.init_range)
        # init_cont(self.hid2output, args.init_range)
        # if args.share_attn:
        #     init_cont(self.attn, args.init_range)
        # else:
        #     init_cont(self.attn, args.init_range)
        #     init_cont(self.lang_attn, args.init_range)
        #     init_cont(self.sel_attn, args.init_range)
        #     init_cont(self.ref_attn, args.init_range)

        if args.is_selection_prediction:
            input_dim = 0
            if args.dot_recurrence and 'is_selection' in args.dot_recurrence_in:
                input_dim += args.dot_recurrence_dim * self.dot_recurrence_embeddings
            if 'language_state' in args.is_selection_prediction_features:
                input_dim += args.nhid_lang
            if 'turn' in args.is_selection_prediction_features:
                input_dim += 1
            if 'dot_context' in args.is_selection_prediction_features:
                input_dim += args.nembed_ctx
            if args.is_selection_prediction_layers == 0:
                self.is_selection_layer = nn.Linear(input_dim, 1)
            else:
                self.is_selection_layer = FeedForward(
                    args.is_selection_prediction_layers, input_dim, 128, 1, dropout_p=args.dropout
                )

        if args.structured_temporal_attention_transitions_language == 'between_mentions':
            self.between_mention_self_attn_layer = nn.Linear(args.nhid_lang * self.num_reader_directions, 1)

    def initialize_state(self, ctx, belief_constructor) -> State:
        ctx_h = self.ctx_encoder(ctx)
        ctx_differences = self.ctx_differences(ctx)

        bsz = ctx_h.size(0)
        reader_and_writer_lang_h = self._init_h(bsz)
        dot_h_maybe_multi = self._init_dot_h_maybe_multi(bsz, False)
        dot_h_maybe_multi_structured = self._init_dot_h_maybe_multi(bsz, True)
        return State(
            self.args, bsz, ctx, ctx_h, ctx_differences, reader_and_writer_lang_h,
            dot_h_maybe_multi, dot_h_maybe_multi_structured, belief_constructor,
            turn=0,
        )

    @property
    def num_reader_directions(self):
        if vars(self.args).get('bidirectional_reader', False):
            return 2
        else:
            return 1

    def reader_forward_hs(self, reader_lang_hs):
        assert reader_lang_hs.dim() == 3
        # batch x seq_len x num_directions x hidden size.
        # or is it seq_len x batch x num_directions x hidden size?
        # should be fine either way
        separated = reader_lang_hs.view(reader_lang_hs.size(0), reader_lang_hs.size(1), self.num_reader_directions, -1)
        return separated[:,:,0]

    def reader_forward_last_h(self, reader_last_h):
        assert reader_last_h.dim() == 3
        num_layers = 1
        assert reader_last_h.size(0) == self.num_reader_directions
        # num_layers(1) x num_directions x batch x hidden_size
        separated = reader_last_h.unsqueeze(0)
        return separated[:,0]

    def _attention_name(self, name):
        return '{}_attn'.format(name)

    def _apply_attention(self, name, lang_input, input, ctx_differences, num_markables, joint_factor_input, lang_between_mentions_input=None, ctx=None):
        if self.args.share_attn:
            return self.attn(lang_input, input, ctx_differences, num_markables, joint_factor_input, lang_between_mentions_input, ctx)
        elif self.args.separate_attn:
            attn_module = getattr(self, self._attention_name(name))
            return attn_module(lang_input, input, ctx_differences, num_markables, joint_factor_input, lang_between_mentions_input, ctx)
        else:
            attn_module = getattr(self, self._attention_name(name))
            return attn_module(lang_input, self.attn_prefix(input), ctx_differences, num_markables, joint_factor_input, lang_between_mentions_input, ctx)

    def _zero(self, *sizes):
        h = torch.Tensor(*sizes).fill_(0)
        return Variable(h)

    def flatten_parameters(self):
        self.reader.flatten_parameters()
        if self.args.untie_grus:
            self.writer.flatten_parameters()

    def embed_dialogue(self, inpt):
        inpt_emb = self.word_embed(inpt)
        inpt_emb = self.dropout(inpt_emb)
        return inpt_emb

    def ctx_differences(self, ctx):
        assymmetric_pairs = vars(self.args).get('structured_attention_asymmetric_pairs', False)
        _, _, ctx_differences = pairwise_differences(
            ctx, num_ent=self.num_ent, dim_ent=4, symmetric=True, relation_include=[],
            include_asymmetric_rep_in_symmetric=assymmetric_pairs
        )
        return ctx_differences

    def _gather_from_inpt(self, temporal_inputs, inpt):
        #inpt: batch_size x num_refs x 3
        assert inpt.size(-1) == 3
        bsz = inpt.size(0)

        # (3 * num_refs) x batch_size
        inpt = torch.transpose(inpt, 0, 2).contiguous().view(-1, bsz)
        # (3 * num_refs) x batch_size x 1
        inpt = inpt.view(-1, bsz).unsqueeze(2)
        # (3 * num_refs) x batch_size x hidden_dim
        inpt = inpt.expand(-1, -1, temporal_inputs.size(2))

        # gather indices
        inpt = torch.gather(temporal_inputs, 0, inpt)
        # reshape
        inpt = inpt.view(3, -1, inpt.size(1), inpt.size(2))
        return inpt

    def _pool_between_referents(self, temporal_inputs, inpt):
        bsz = temporal_inputs.size(1)
        device = temporal_inputs.device
        pooled = []
        if inpt.size(1) > 1:
            # T x bsz
            attn_logits = self.between_mention_self_attn_layer(temporal_inputs).squeeze(-1)
            start_posts = inpt[:,:-1,0]
            end_posts = inpt[:,1:,1]
            for pair_index in range(start_posts.size(1)):
                mask = torch.zeros(temporal_inputs.size()[:2]).to(device).bool()
                for bix in range(bsz):
                    start = start_posts[bix,pair_index]
                    end = end_posts[bix,pair_index]
                    mask[start:end+1,bix] = True
                masked_attn_logits = torch.where(mask, attn_logits, torch.tensor(-1e9).to(device))
                attention = masked_attn_logits.softmax(0)
                this_pooled = torch.einsum("tb,tbd->bd", attention, temporal_inputs)
                pooled.append(this_pooled)
        return pooled

    def reference_resolution(self, state: _State, outs_emb, ref_inpt, num_markables, for_self=True,
                             ref_beliefs=None):
        # ref_inpt: bsz x num_refs x 3
        if ref_inpt is None:
            return None

        ctx_h = state.ctx_h
        ctx_differences = state.ctx_differences

        bsz = ctx_h.size(0)
        num_dots = ctx_h.size(1)

        if ref_beliefs is not None:
            ctx_h = torch.cat((ctx_h, ref_beliefs), -1)

        # reshape
        emb_gathered = self._gather_from_inpt(outs_emb, ref_inpt)

        if vars(self.args).get('structured_temporal_attention_transitions_language', 'subtract_mentions') == 'between_mentions':
            emb_between = self._pool_between_referents(outs_emb, ref_inpt)
        else:
            emb_between = None

        # pool embeddings for the referent's start and end, as well as the end of the sentence it occurs in
        # to produce ref_inpt of size
        # num_refs x batch_size x hidden_dim
        if vars(self.args).get('learned_pooling', False):
            # learned weights
            ref_weights = (self.ref_pooling_weights if for_self else self.partner_ref_pooling_weights).softmax(-1)
            emb_gathered = torch.einsum("tnbh,t->nbh", (emb_gathered, ref_weights))
        else:
            # mean pooling
            emb_gathered = torch.mean(emb_gathered, 0)

        # num_refs x batch_size x num_dots x hidden_dim
        emb_gathered_expand = emb_gathered.unsqueeze(2).expand(-1, -1, num_dots, -1)
        ctx_h = ctx_h.unsqueeze(0).expand(
            emb_gathered_expand.size(0), emb_gathered_expand.size(1), emb_gathered_expand.size(2), ctx_h.size(-1)
        )

        if vars(self.args).get('partner_reference_prediction', False):
            attention_params_name = 'ref' if for_self else 'ref_partner'
        else:
            attention_params_name = 'ref'
        outs = self._apply_attention(
            attention_params_name, emb_gathered, torch.cat([emb_gathered_expand, ctx_h], -1), ctx_differences, num_markables,
            joint_factor_input=state.dot_h_maybe_multi_structured, lang_between_mentions_input=emb_between,
            ctx=state.ctx,
        )
        return outs

    def next_mention_latents(self, state: _State, outs_emb, lens, mention_beliefs, mention_latent_beliefs,
                             dots_mentioned_num_markables_to_force=None, min_num_mentions=0, max_num_mentions=12, can_confirm=None):
        ctx_h = state.ctx_h
        ctx_differences = state.ctx_differences
        bsz = ctx_h.size(0)
        num_dots = ctx_h.size(1)

        if mention_beliefs is not None:
            ctx_h = torch.cat((ctx_h, mention_beliefs), -1)

        if vars(self.args).get('next_mention_prediction_no_lang', False):
            lang_states = self.next_mention_start_emb.unsqueeze(0).repeat_interleave(bsz, dim=0)
        else:
            # 1 x batch_size x hidden_dim
            lens_expand = lens.unsqueeze(0).unsqueeze(2).expand(-1, -1, outs_emb.size(2))
            # 1 x batch_size x hidden_dim
            latent_states = torch.gather(outs_emb, 0, lens_expand-1)
            # batch_size x hidden_dim
            lang_states = latent_states.squeeze(0)

        if self.args.next_mention_prediction_type == 'multi_reference':
            is_finished = torch.zeros_like(lens).bool()
            dots_mentioned_num_markables = torch.zeros_like(lens).long()
            input_context = lang_states
            if mention_latent_beliefs is not None:
                mention_latent_beliefs_pooled = mention_latent_beliefs.mean(1)
                input_context = torch.cat((input_context, mention_latent_beliefs_pooled), -1)
            h = self.next_mention_transform(input_context)
            if vars(self.args).get('hidden_context_confirmations') and 'next_mention' in vars(self.args).get('hidden_context_confirmations_in', ['generation']) and can_confirm is not None:
                # assert can_confirm is not None
                sig_weight = self.hidden_ctx_confirmations_nm_weight.sigmoid()
                conf_embs = self.hidden_ctx_confirmations_nm_embeddings(can_confirm.long())
                h = (1-sig_weight) * h + sig_weight * conf_embs
            hs = []
            # stop_losses = torch.zeros_like(lens).float()
            stop_losses = []
            # dict lookup so we throw an error if a bad argument is passed
            predict_stop_at_each_position = {
                'bernoulli': True,
                'categorical': False,
            }[vars(self.args).get('next_mention_length_prediction_type', 'bernoulli')]
            if predict_stop_at_each_position:
                predicted_lengths = None
            else:
                predicted_length_logits = self.next_mention_length(input_context)
                predicted_lengths = predicted_length_logits.argmax(-1)
                if dots_mentioned_num_markables_to_force is not None:
                    stop_losses.append(self.next_mention_length_crit(predicted_length_logits, dots_mentioned_num_markables_to_force))
                else:
                    stop_losses.append(self.next_mention_length_crit(predicted_length_logits, predicted_lengths))
            while not is_finished.all():
                if predict_stop_at_each_position:
                    stop_logits = self.next_mention_stop(h).squeeze(-1)
                else:
                    stop_logits = None
                if dots_mentioned_num_markables_to_force is not None:
                    should_stop = dots_mentioned_num_markables >= dots_mentioned_num_markables_to_force
                    # stop_logit > 0: don't stop
                    # stop_logit < 0: stop
                    if predict_stop_at_each_position:
                        stop_losses.append(
                            -1.0 * (stop_logits * (should_stop.float() * 2 - 1)).sigmoid().log() * (~is_finished).float()
                        )
                    is_finished = dots_mentioned_num_markables >= dots_mentioned_num_markables_to_force
                else:
                    if predict_stop_at_each_position:
                        stop_losses.append(
                            -1.0 * stop_logits.abs().sigmoid().log() * (~is_finished).float()
                        )
                        # stop_losses.append(torch.zeros(bsz).to(stop_logits.device))
                        is_finished |= (stop_logits > 0)
                    else:
                        is_finished |= dots_mentioned_num_markables >= predicted_lengths
                is_finished |= (dots_mentioned_num_markables >= max_num_mentions)
                if min_num_mentions > 0:
                    # if num_markables < min_num_mentions then is_finished = False
                    is_finished &= dots_mentioned_num_markables >= min_num_mentions
                dots_mentioned_num_markables += (~is_finished).long()
                if not is_finished.all():
                    h = self.next_mention_cell(lang_states, h)
                    hs.append(h)
            assert len(hs) == dots_mentioned_num_markables.max().item()
            if dots_mentioned_num_markables_to_force is not None:
                assert (dots_mentioned_num_markables == dots_mentioned_num_markables_to_force).all()
            if len(stop_losses) != 0:
                stop_losses = torch.stack(stop_losses, 0)
            else:
                stop_losses = torch.zeros((1, bsz), requires_grad=True).to(latent_states.device)
            if hs:
                latent_states = torch.stack(hs, dim=0)
            else:
                latent_states = torch.zeros((0, bsz, lang_states.size(-1)))
            ctx_h = ctx_h.unsqueeze(0).repeat_interleave(latent_states.size(0), dim=0)
        elif self.args.next_mention_prediction_type == 'collapsed':
            # add a dummy time dimension for attention
            # 1 x batch_size x num_dots x hidden_dim
            ctx_h = ctx_h.unsqueeze(0)
            latent_states = lang_states.unsqueeze(0)
            dots_mentioned_num_markables = torch.full((bsz,), 1).long().to(latent_states.device)
            stop_losses = torch.zeros((1, bsz), requires_grad=True).to(latent_states.device)
        else:
            raise ValueError(f"--next_mention_prediction_type={self.args.next_mention_prediction}")

        # T x batch_size x num_dots x hidden_dim
        return NextMentionLatents(
            latent_states=latent_states, dots_mentioned_num_markables=dots_mentioned_num_markables, stop_losses=stop_losses, ctx_h_with_beliefs=ctx_h
        )

    def next_mention_prediction_from_latents(self, state: State, latents: NextMentionLatents):
        ctx_h = state.ctx_h
        ctx_differences = state.ctx_differences
        num_dots = ctx_h.size(1)
        # sum over time dimension to get a vector of size bsz
        stop_loss = latents.stop_losses.sum(0)
        states_expand = latents.latent_states.unsqueeze(2).expand(-1, -1, num_dots, -1)
        if (latents.dots_mentioned_num_markables > 0).any():
            scores = self._apply_attention(
                'next_mention',
                latents.latent_states,
                torch.cat([states_expand, latents.ctx_h_with_beliefs], -1), ctx_differences, latents.dots_mentioned_num_markables,
                joint_factor_input=state.dot_h_maybe_multi_structured,
                ctx=state.ctx,
            )
        else:
            scores = None
        return scores, stop_loss, latents.dots_mentioned_num_markables

    def next_mention_prediction(self, state: State, outs_emb, lens, mention_beliefs, mention_latent_beliefs,
                                dots_mentioned_num_markables_to_force=None, min_num_mentions=0, max_num_mentions=12,
                                can_confirm=None):
        latents = self.next_mention_latents(
            state, outs_emb, lens, mention_beliefs, mention_latent_beliefs, dots_mentioned_num_markables_to_force,
            min_num_mentions, max_num_mentions, can_confirm=can_confirm
        )
        return self.next_mention_prediction_from_latents(state, latents)

    def selection(self, state: _State, outs_emb, sel_idx, beliefs=None):
        # outs_emb: length x batch_size x dim
        ctx_h = state.ctx_h
        bsz = outs_emb.size(1)
        ctx_differences = state.ctx_differences

        if vars(self.args).get('selection_prediction_no_lang', False):
            sel_inpt = self.selection_start_emb.unsqueeze(0).repeat_interleave(bsz, dim=0)
        else:
            if self.args.selection_attention:
                # batch_size x entity_dim
                transformed_ctx = self.ctx_layer(ctx_h).mean(1)
                # length x batch_size x dim
                #
                text_and_transformed_ctx = torch.cat([
                    transformed_ctx.unsqueeze(0).expand(outs_emb.size(0), -1, -1),
                    outs_emb], dim=-1)
                # TODO: add a mask
                # length x batch_size
                attention_logits = self.selection_attention_layer(text_and_transformed_ctx).squeeze(-1)
                attention_probs = attention_logits.softmax(dim=0)
                sel_inpt = torch.einsum("lb,lbd->bd", [attention_probs,outs_emb])
            else:
                sel_idx = sel_idx.unsqueeze(0)
                sel_idx = sel_idx.unsqueeze(2)
                sel_idx = sel_idx.expand(sel_idx.size(0), sel_idx.size(1), outs_emb.size(2))
                sel_inpt = torch.gather(outs_emb, 0, sel_idx)
                sel_inpt = sel_inpt.squeeze(0)
            #  batch_size x hidden
        sel_inpt = sel_inpt.unsqueeze(1)

        # stack alongside the entity embeddings
        sel_inpt_expand = sel_inpt.expand(-1, ctx_h.size(1), -1)
        to_cat = [sel_inpt_expand, ctx_h]
        if beliefs is not None:
            to_cat.append(beliefs)
        # TODO: pass something for num_markables for consistency; right now it relies on selection not using StructuredTemporalAttention
        return self._apply_attention(
            'sel', sel_inpt, torch.cat(to_cat, 2), ctx_differences, num_markables=None,
            joint_factor_input=state.dot_h_maybe_multi_structured,
            ctx=state.ctx,
        )

    def _language_conditioned_dot_attention(self, ctx_differences, ctx_h, lang_hs, attention_type,
                                            dots_mentioned, dots_mentioned_per_ref, generation_beliefs,
                                            structured_generation_beliefs=None, ctx=None):
        # lang_hs: seq_len x batch_size x hidden
        # ctx_h: batch_size x num_dots x nembed_ctx
        # dots_mentioned: batch_size x num_dots, binary tensor of which dots to allow attention on. if all zero, output zeros
        assert ctx_h.dim() == 3
        assert lang_hs.dim() == 3

        seq_len, bsz, _ = lang_hs.size()
        bsz_, num_dots, _ = ctx_h.size()
        assert bsz == bsz_, (bsz, bsz_)

        # seq_len = lang_hs.size(0)
        # bsz = lang_hs.size(1)
        # num_dots = ctx_h.size(1)

        assert attention_type in ['feed', 'lang', 'hidden']
        # I messed up the argument naming
        # TODO: maybe rename "lang" -> "word" everywhere
        if attention_type == 'lang':
            constrain_attention_type = 'word'
        else:
            constrain_attention_type = attention_type
        constrain_attention = vars(self.args).get('{}_attention_constrained'.format(constrain_attention_type), False)

        # if use_feed_attn and vars(self.args).get('feed_attention_constrained', False):
        #     constrain_attention = True
        # if (not use_feed_attn) and vars(self.args).get('word_attention_constrained', False):
        #     constrain_attention = True

        if constrain_attention:
            assert dots_mentioned is not None

        if dots_mentioned is not None:
            bsz_, num_dots_ = dots_mentioned.size()
            assert bsz == bsz_, (bsz, bsz_)
            assert num_dots == num_dots_, (num_dots, num_dots_)

        # expand num_ent dimensions to calculate attention scores
        # seq_len x batch_size x num_dots x _
        lang_h_expand = lang_hs.unsqueeze(2).expand(-1, -1, num_dots, -1)

        # seq_len x batch_size x num_dots x nembed_ctx
        if vars(self.args).get('mark_dots_mentioned', False):
            # TODO: consider tiling this indicator feature across more dimensions
            assert dots_mentioned is not None
            ctx_h_marked = torch.cat((ctx_h, dots_mentioned.float().unsqueeze(-1)), -1)
        else:
            ctx_h_marked = ctx_h
        if generation_beliefs is not None:
            ctx_h_marked = torch.cat((ctx_h_marked, generation_beliefs), dim=-1)
        ctx_h_expand = ctx_h_marked.unsqueeze(0).expand(seq_len, -1, -1, -1)

        # seq_len x batch_size x num_dots
        attn_logit, _, _ = self._apply_attention(
            attention_type,
            lang_hs,
            torch.cat([lang_h_expand, ctx_h_expand], 3),
            ctx_differences,
            num_markables=None,
            joint_factor_input=structured_generation_beliefs,
            ctx=ctx,
        )

        if constrain_attention:
            attn_logit = attn_logit.masked_fill(~dots_mentioned.unsqueeze(0).expand_as(attn_logit), BIG_NEG)

        # language-conditioned attention over the dots
        # seq_len x batch_size x num_dots
        attention_type = vars(self.args).get('attention_type', 'softmax')
        if attention_type == 'softmax':
            attn_prob = F.softmax(attn_logit, dim=2)
        elif attention_type == 'sigmoid':
            attn_prob = torch.sigmoid(attn_logit)
        else:
            raise ValueError("invalid --attention_type: {}".format(attention_type))

        if constrain_attention:
            zero_rows = (dots_mentioned.sum(dim=-1) == 0)
            attn_prob = attn_prob.masked_fill(zero_rows.unsqueeze(0).unsqueeze(-1).expand_as(attn_prob), 0.0)

        # attn_prob[:] = 0
        # attn_prob[:,:,0] = 1.0

        # from IPython.core.debugger import set_trace
        # set_trace()

        return attn_prob

    def make_l1_scoring_function(
        self, state: _State, inpt, tgt, ref_inpt,
        num_markables, partner_num_markables,
        lens, belief_constructor=None,
        partner_ref_inpt=None, timestep=0, partner_ref_outs=None, ref_outs=None,
        next_mention_out=None,
    ):
        ctx = state.ctx
        ctx_h = state.ctx_h
        ctx_differences = state.ctx_differences
        reader_and_writer_lang_h = state.reader_and_writer_lang_h
        # sel_idx isn't needed b/c we'll pass compute_sel_idx = False
        crit_no_reduce = Criterion(self.word_dict, bad_toks=['<pad>'], reduction='none')

        def score_refs(candidate_dots_mentioned_per_ref, normalize_over_candidates=False):
            # candidate_dots_mentioned: num_mentions x batch_size x num_candidates x num_dots
            num_mentions, bsz, num_candidates, num_dots = candidate_dots_mentioned_per_ref.size()
            T = inpt.size(0)

            def comb(tensor, batch_dim):
                    # combine batch and candidate dims
                    new_dims = tensor.size()[:batch_dim] + (tensor.size(batch_dim) * tensor.size(batch_dim+1),) + tensor.size()[batch_dim+2:]
                    return tensor.view(new_dims)

            def tile(tensor, batch_dim=0):
                if tensor is None:
                    return None
                assert tensor.size(batch_dim) == bsz
                # add candidate dimension at batch_dim+1, then combine it with the batch dim
                return comb(
                    torch.repeat_interleave(tensor.unsqueeze(batch_dim+1), repeats=num_candidates, dim=batch_dim+1),
                    batch_dim
                )

            def uncomb(tensor, batch_dim=0):
                new_dims = tensor.size[:batch_dim] + (bsz, num_candidates) + tensor.size[batch_dim+1:]
                return tensor.view(new_dims)

            def tile_state(state: _State):
                reader_lang_h, writer_lang_h = state.reader_and_writer_lang_h
                reader_lang_h_t = tile(reader_lang_h, batch_dim=1)
                writer_lang_h_t = tile(writer_lang_h, batch_dim=1)
                reader_and_writer_lang_h_t = (reader_lang_h_t, writer_lang_h_t)
                return state._replace(
                    ctx=tile(state.ctx),
                    ctx_differences=tile(state.ctx_differences),
                    ctx_h=tile(state.ctx_h),
                    reader_and_writer_lang_h=reader_and_writer_lang_h_t
                )

            inpt_t = tile(inpt, batch_dim=1)
            tgt_t = tile(tgt.view(T, bsz), batch_dim=1).flatten()
            ref_inpt_t = tile(ref_inpt)
            lens_t = tile(lens)

            partner_ref_inpt_t = tile(partner_ref_inpt)
            num_markables_t = tile(num_markables)
            partner_num_markables_t = tile(partner_num_markables)
            # marginal and full distributions
            ref_outs_t = [(tile(t[0], batch_dim=1), tile(t[1], batch_dim=1))
                          if t is not None else None
                          for t in ref_outs]
            if partner_ref_outs is not None:
                partner_ref_outs_t = [(tile(t[0], batch_dim=1), tile(t[1], batch_dim=1))
                                      if t is not None else t
                                      for t in partner_ref_outs]
            else:
                partner_ref_outs_t = None

            # mention_scores = torch.zeros(candidate_dots_mentioned.size()[:-1])
            s0_probs_per_mention = []
            if self.args.only_first_mention:
                raise NotImplementedError("only_first_mention for ref scoring isn't implemented yet")

            # TODO: do we really need triply-nested functions?
            def get_log_probs(this_mentions, this_mentions_per_ref):
                this_mentions_t = comb(this_mentions, batch_dim=0)
                this_mentions_per_ref_t = comb(this_mentions_per_ref, batch_dim=0)

                raise NotImplementedError("dot h, can_confirm, can_confirm_next")

                state, outs, *_ = self._forward(
                    state_t, inpt_t, lens_t,
                    ref_inpt=ref_inpt_t, partner_ref_inpt=partner_ref_inpt_t,
                    compute_sel_out=False, sel_idx=None,
                    num_markables=num_markables_t,
                    partner_num_markables=partner_num_markables_t,
                    dots_mentioned=this_mentions_t,
                    dots_mentioned_per_ref=this_mentions_per_ref_t,
                    belief_constructor=belief_constructor, timestep=timestep,
                    partner_ref_outs=partner_ref_outs_t[:timestep] if partner_ref_outs_t is not None else None,
                    ref_outs=ref_outs_t[:timestep] if ref_outs_t is not None else None,
                )
                log_probs = -crit_no_reduce(outs, tgt_t).view(T, bsz, num_candidates).sum(0)
                return log_probs

            # candidate_dots_mentioned: max_num_mentions x bsz x num_candidates x num_dots
            candidate_dots_mentioned_per_ref_rearrange = einops.rearrange(candidate_dots_mentioned_per_ref, "nm b c d->b c nm d")

            candidate_dots_mentioned = (candidate_dots_mentioned_per_ref.sum(0) > 0)
            unnormed_l1_probs = get_log_probs(candidate_dots_mentioned, candidate_dots_mentioned_per_ref_rearrange)

            l1_prior = vars(self.args).get('l1_prior', 'uniform')
            if l1_prior == 'next_mention':
                assert self.args.next_mention_prediction
                if self.args.max_mentions_in_generation_training != 1:
                    raise NotImplementedError("need to implement a hierarchical next-mention model for the prior")
                next_mention_out_logits, next_mention_out_full, _ = next_mention_out
                if next_mention_out_full is None:
                    # TODO: refactor this to move logits_to_full elsewhere
                    predictor = PragmaticReferencePredictor(self.args)
                    # 1 x bsz x 2 x ...
                    next_mention_out_full = predictor.marginal_logits_to_full_logits(next_mention_out_logits).contiguous()
                # \log p(d). squeeze to remove the *num mention* dimension

                # TODO: deal with a hierarchical next_mention model which predicts distributions over multiple mentions
                # bsz x 2^7
                prior_probs = next_mention_out_full.view(1, next_mention_out_full.size(1), -1).squeeze(0)
                # detach to not backprop through the prior p(d)
                prior_probs = prior_probs.detach()

                # bsz x num_candidates x 2^7
                candidate_dots_mentioned = candidate_dots_mentioned_per_ref.sum(0) > 0
                # bsz x num_candidates
                candidate_dots_mentioned_indices = bit_to_int_array(candidate_dots_mentioned)

                # select values for the candidates
                # bsz x num_candidates
                candidate_prior_probs = prior_probs.gather(-1, candidate_dots_mentioned_indices)

                # \log p(u | d) + \log p(d)
                unnormed_l1_probs += candidate_prior_probs
            elif l1_prior == 'uniform':
                pass
            else:
                raise ValueError(f"invalid --l1_prior={self.args.l1_prior}")
            if normalize_over_candidates:
                return unnormed_l1_probs.log_softmax(-1)
            else:
                return unnormed_l1_probs

        return score_refs

    def _update_dot_h_single(self, dot_h, reader_lang_hs, ref_inpt, num_markables, ref_out, ref_tgt,
                             pooling_weights, transform, cell, structured, oracle_ref_tgts):
        # TODO: implement structured
        dot_h = dot_h.clone()
        indices_to_update = num_markables > 0
        # predictor.forward(ref_inpt[indices_to_update], ref)
        # 3 x max_num_mentions x filtered_bsz x hidden
        h_gathered = self._gather_from_inpt(reader_lang_hs[:,indices_to_update], ref_inpt[indices_to_update])
        # max_num_mentions x filtered_bsz x hidden
        h_pooled = torch.einsum("tnbh,t->nbh", (h_gathered, pooling_weights.softmax(-1)))

        input_names = vars(self.args).get('dot_recurrence_inputs', ['weighted_hidden'])

        # todo: use the structured distribution rather than the dot marginals?
        ref_marginal_logits, ref_joint_logits, _ = ref_out

        if 'predicted_hidden' in input_names or 'predicted_average' in input_names or 'predicted_max' in input_names:
            assert self.args.detach_beliefs
            predictor = ReferencePredictor(self.args)
            _, ref_pred_full, _ = predictor.forward(ref_inpt, ref_tgt, ref_out, num_markables)
            ref_pred = ref_pred_full[:,indices_to_update].float()
        else:
            ref_pred = None

        if vars(self.args).get('dot_recurrence_mention_attention'):
            # max_num_mentions x filtered_bsz
            mention_weights = self.dot_recurrence_mention_attention(h_pooled).squeeze(-1).softmax(0)
        else:
            # max_num_mentions x filtered_bsz
            mention_weights = torch.zeros((h_pooled.size(0), h_pooled.size(1)), device=h_pooled.device)
            for i, nm in enumerate(num_markables[indices_to_update]):
                mention_weights[:nm, i] = 1
            mention_weights /= mention_weights.sum(0, keepdim=True).clamp_min(1)

        uniform_over_dots = vars(self.args).get('dot_recurrence_uniform')

        # max_num_mentions x filtered_bsz x num_dots
        if oracle_ref_tgts:
            # ref_tgt: bsz x max_num_mentions x num_dots
            # attention_probs: max_num_mentions x filtered_bsz x num_dots
            attention_probs = ref_tgt[indices_to_update].transpose(0,1).float()
            assert not structured
            assert not uniform_over_dots
        elif uniform_over_dots:
            attention_probs = torch.ones_like(ref_marginal_logits[:,indices_to_update])
        else:
            if structured:
                attention_probs = ref_joint_logits[:,indices_to_update]
                if self.dot_recurrence_weight_temperature != 1.0:
                    attention_probs /= self.dot_recurrence_weight_temperature
                attention_probs = attention_probs.view(attention_probs.size(0), attention_probs.size(1), -1).softmax(-1)
            else:
                attention_probs = ref_marginal_logits[:,indices_to_update]
                if self.dot_recurrence_weight_temperature != 1.0:
                    attention_probs /= self.dot_recurrence_weight_temperature
                attention_probs = attention_probs.sigmoid()
            if self.args.detach_beliefs:
                attention_probs = attention_probs.detach()

        to_cat = []
        if 'weighted_hidden' in input_names:
            to_cat.append(torch.einsum("nbh,nbd,nb->bdh", (h_pooled, attention_probs, mention_weights)))
        if 'weights_average' in input_names:
            to_cat.append(torch.einsum("nbd,nb->bd", (attention_probs, mention_weights)).unsqueeze(-1))
        if 'weights_max' in input_names:
            to_cat.append(einops.reduce(attention_probs, "n b d -> b d", 'max').unsqueeze(-1))
        if 'predicted_hidden' in input_names:
            to_cat.append(torch.einsum("nbh,nbd,nb->bdh", (h_pooled, ref_pred, mention_weights)))
        if 'predicted_average' in input_names:
            to_cat.append(torch.einsum("nbd,nb->bd", (ref_pred, mention_weights)).unsqueeze(-1))
        if 'predicted_max' in input_names:
            to_cat.append(einops.reduce(ref_pred, "n b d -> b d", 'max').unsqueeze(-1))

        inputs = einops.rearrange(torch.cat(to_cat, -1), "b d h -> (b d) h")
        transformed_inputs = transform(inputs)
        dot_h_flat = einops.rearrange(dot_h[indices_to_update], "b d h -> (b d) h")
        dot_h_update = cell(transformed_inputs, dot_h_flat)
        dot_h_update = einops.rearrange(dot_h_update, "(b d) h -> b d h", b=indices_to_update.sum())
        dot_h[indices_to_update] = dot_h_update
        return dot_h

    def _init_dot_h_maybe_multi(self, bsz, structured):
        def _repeat(dot_h):
            dot_h = dot_h.unsqueeze(0).repeat_interleave(bsz, dim=0)
            if structured:
                dot_h = dot_h.unsqueeze(1).repeat_interleave(2**self.num_ent, dim=1)
            else:
                dot_h = dot_h.unsqueeze(1).repeat_interleave(self.num_ent, dim=1)
            return dot_h
        if vars(self.args).get('dot_recurrence', False):
            # bsz x num_dots x hidden
            if structured and not vars(self.args).get('dot_recurrence_structured', False):
                return None
            dot_h = _repeat(self.dot_rec_structured_init if structured else self.dot_rec_init)
            if vars(self.args).get('dot_recurrence_split'):
                dot_partner_h = _repeat(self.dot_rec_partner_structured_init if structured else self.dot_rec_partner_init)
                return dot_h, dot_partner_h
            else:
                return dot_h
        else:
            return None

    def _update_dot_h_maybe_multi(self, state: _State, reader_lang_hs,
                                  ref_inpt, partner_ref_inpt,
                                  num_markables, partner_num_markables,
                                  ref_out, partner_ref_out,
                                  ref_tgt, partner_ref_tgt):
        if not self.args.dot_recurrence:
            return state

        # TODO: make this not ugly
        structured_and_fields = [
            (False, 'dot_h_maybe_multi')
        ]
        if vars(self.args).get('dot_recurrence_structured', False):
            structured_and_fields.append((True, 'dot_h_maybe_multi_structured'))

        is_split = vars(self.args).get('dot_recurrence_split')

        for structured, field in structured_and_fields:
            if is_split:
                dot_h, partner_dot_h = getattr(state, field)
            else:
                dot_h = getattr(state, field)

            if ref_out is not None and partner_ref_out is not None:
                assert not ((num_markables > 0) & (partner_num_markables > 0)).any()
            # ref_out can be none if there are no markables in the batch
            if 'self' in self.args.dot_recurrence:
                if ref_out is not None:
                    oracle_refs = vars(self.args).get('dot_recurrence_oracle', False) and \
                                  'self' in vars(self.args).get('dot_recurrence_oracle_for', ['self', 'partner'])
                    dot_h = self._update_dot_h_single(
                        dot_h, reader_lang_hs, ref_inpt, num_markables, ref_out, ref_tgt,
                        self.dot_rec_ref_pooling_weights, self.dot_rec_ref_transform,
                        self.dot_rec_cell, structured, oracle_refs
                    )
            if 'partner' in self.args.dot_recurrence:
                assert self.args.partner_reference_prediction
                # partner_ref_out can be none if there are no markables in the batch
                if partner_ref_out is not None:
                    oracle_refs = vars(self.args).get('dot_recurrence_oracle', False) and \
                                  'partner' in vars(self.args).get('dot_recurrence_oracle_for', ['self', 'partner'])
                    partner_dot_h = self._update_dot_h_single(
                        partner_dot_h if is_split else dot_h, reader_lang_hs, partner_ref_inpt, partner_num_markables,
                        partner_ref_out, partner_ref_tgt,
                        self.dot_rec_partner_ref_pooling_weights, self.dot_rec_partner_ref_transform,
                        self.dot_rec_cell if (not is_split) else self.dot_rec_partner_cell,
                        structured, oracle_refs
                    )
                    if not is_split:
                        dot_h = partner_dot_h
            if is_split:
                dot_h_maybe_multi = (dot_h, partner_dot_h)
            else:
                dot_h_maybe_multi = dot_h
            state = state._replace(**{field: dot_h_maybe_multi})
        return state

    def language_output(self, state: _State, writer_lang_hs, dots_mentioned, dots_mentioned_per_ref,
                        generation_beliefs, ctx_seq_encoded_and_mask):
        # compute language output
        if vars(self.args).get('no_word_attention', False):
            outs = self.hid2output(writer_lang_hs)
            ctx_attn_prob = None
        else:
            ctx_attn_prob = self._language_conditioned_dot_attention(
                state.ctx_differences, state.ctx_h, writer_lang_hs, attention_type='lang',
                dots_mentioned=dots_mentioned,
                dots_mentioned_per_ref=dots_mentioned_per_ref,
                generation_beliefs=generation_beliefs,
                structured_generation_beliefs=state.dot_h_maybe_multi_structured,
                ctx=state.ctx,
            )
            ctx_h_to_expand = state.ctx_h
            if vars(self.args).get('marks_in_word_prediction', False):
                to_cat = [state.ctx_h]
                if vars(self.args).get('mark_dots_mentioned', False):
                    assert dots_mentioned is not None
                    to_cat.append(dots_mentioned.float().unsqueeze(-1))
                if generation_beliefs is not None:
                    to_cat.append(generation_beliefs)
                if len(to_cat) > 1:
                    ctx_h_to_expand = torch.cat(to_cat, dim=-1)
                else:
                    ctx_h_to_expand = to_cat[0]
            ctx_h_expand = ctx_h_to_expand.unsqueeze(0).expand(writer_lang_hs.size(0), -1, -1, -1)

            ctx_h_lang = torch.einsum("tbnd,tbn->tbd", (ctx_h_expand,ctx_attn_prob))
            # ctx_h_lang = torch.sum(torch.mul(ctx_h_expand, lang_prob.unsqueeze(-1)), 2)

            to_cat = [writer_lang_hs, ctx_h_lang]

            if vars(self.args).get('hidden_context_mention_encoder_attention', False):
                encoded, mask = ctx_seq_encoded_and_mask
                if encoded.size(1) == 0:
                    to_cat.append(self.mention_encoder_attention_no_mentions_emb.unsqueeze(0).unsqueeze(1).expand(
                        writer_lang_hs.size(0), writer_lang_hs.size(1), -1
                    ))
                else:
                    # T x bsz x max_num_mentions x 1
                    # attention_logits = torch.einsum(
                    #     "tbx,bny,oyx->tbno", (writer_lang_hs, encoded, self.mention_encoder_attention.weight)
                    # )
                    attention_logits = self.mention_encoder_attention(
                        encoded.unsqueeze(0).expand(writer_lang_hs.size(0), -1, -1, -1).contiguous(),
                        writer_lang_hs.unsqueeze(2).expand(-1, -1, encoded.size(1), -1).contiguous(),
                    )
                    attention_logits = attention_logits.squeeze(-1)
                    # T x bsz x max_num_mentions [normalized over mentions]
                    attention_probs = attention_logits.softmax(2)
                    weighted_encoded = torch.einsum(
                        "bny,tbn->tby", (encoded, attention_probs)
                    )
                    to_cat.append(weighted_encoded)

            outs = self.hid2output(torch.cat(to_cat, -1))
        outs = F.linear(outs, self.word_embed.weight)
        outs = outs.view(-1, outs.size(-1))
        return outs, ctx_attn_prob

    def is_selection_prediction(self, state: State):
        _, writer_lang_h = state.reader_and_writer_lang_h
        bsz = writer_lang_h.size(1)
        num_dots = state.ctx_h.size(1)
        device = writer_lang_h.device
        if not vars(self.args).get('is_selection_prediction', False):
            return torch.zeros(bsz).to(device)
        to_cat = []
        if 'language_state' in self.args.is_selection_prediction_features:
            # bsz x num_dots x hidden
            to_cat.append(writer_lang_h.squeeze(0).unsqueeze(1).repeat_interleave(num_dots, dim=1))
        if self.args.dot_recurrence and 'is_selection' in self.args.dot_recurrence_in:
            to_cat.append(state.dot_h())
        if 'turn' in self.args.is_selection_prediction_features:
            turn_scalar = torch.full((bsz, num_dots, 1), state.turn).float().to(device)
            to_cat.append(turn_scalar / 10)
        if 'dot_context' in self.args.is_selection_prediction_features:
            to_cat.append(state.ctx_h)
        inputs = torch.cat(to_cat, dim=-1)
        logits = self.is_selection_layer(inputs).max(1).values.squeeze(-1)
        return logits

    def _forward(self, state: State, inpt, lens,
                 ref_inpt=None, partner_ref_inpt=None,
                 num_markables=None, partner_num_markables=None,
                 compute_sel_out=False, sel_idx=None,
                 dots_mentioned=None, dots_mentioned_per_ref=None,
                 dots_mentioned_num_markables=None,
                 timestep=0,
                 # needed for generation beliefs
                 ref_outs=None, partner_ref_outs=None,
                 # needed for dot_recurrence_oracle
                 ref_tgt=None, partner_ref_tgt=None,
                 force_next_mention_num_markables=False,
                 next_dots_mentioned_num_markables=None,
                 is_selection=None,
                 can_confirm=None,
                 can_confirm_next=None,
                 ):
        # ctx_h: bsz x num_dots x nembed_ctx
        # lang_h: num_layers*num_directions x bsz x nhid_lang
        # dot_h: None or bsz x num_dots x dot_recurrence_dim
        ctx = state.ctx
        ctx_h = state.ctx_h
        ctx_differences = state.ctx_differences
        bsz = ctx_h.size(0)
        seq_len = inpt.size(0)

        # dots_mentioned_per_ref: bsz x max_num_mentions x num_dots

        # print('inpt size: {}'.format(inpt.size()))

        generation_beliefs = state.make_beliefs('generation', timestep, partner_ref_outs, ref_outs)

        (reader_lang_hs, writer_lang_hs), state, feed_ctx_attn_prob, ctx_seq_encoded_and_mask = self._read(
            state, inpt, lens,
            pack=True,
            dots_mentioned=dots_mentioned,
            dots_mentioned_per_ref=dots_mentioned_per_ref,
            dots_mentioned_num_markables=dots_mentioned_num_markables,
            generation_beliefs=generation_beliefs,
            is_selection=is_selection,
            can_confirm=can_confirm,
        )

        outs, ctx_attn_prob = self.language_output(
            state, writer_lang_hs, dots_mentioned, dots_mentioned_per_ref, generation_beliefs,
            ctx_seq_encoded_and_mask
        )

        ref_beliefs = state.make_beliefs('ref', timestep, partner_ref_outs, ref_outs)
        partner_ref_beliefs = state.make_beliefs('partner_ref', timestep, partner_ref_outs, ref_outs)

        ref_out = self.reference_resolution(
            state, reader_lang_hs, ref_inpt, for_self=True, ref_beliefs=ref_beliefs,
            num_markables=num_markables,
        )

        if vars(self.args).get('partner_reference_prediction', False):
            partner_ref_out = self.reference_resolution(
                state, reader_lang_hs, partner_ref_inpt, for_self=False, ref_beliefs=partner_ref_beliefs,
                num_markables=partner_num_markables,
            )
        else:
            partner_ref_out = None

        if self.args.dot_recurrence:
            state = self._update_dot_h_maybe_multi(state, reader_lang_hs,
                                                   ref_inpt, partner_ref_inpt,
                                                   num_markables, partner_num_markables,
                                                   ref_out, partner_ref_out,
                                                   ref_tgt, partner_ref_tgt)

        if vars(self.args).get('next_mention_prediction', False):
            assert lens is not None
            mention_beliefs = state.make_beliefs(
                'mention', timestep, partner_ref_outs + [partner_ref_out], ref_outs + [ref_out]
            )
            mention_latent_beliefs = state.make_beliefs(
                'next_mention_latents', timestep, partner_ref_outs + [partner_ref_out], ref_outs + [ref_out]
            )
            # TODO: consider using reader_lang_hs for this
            next_mention_latents = self.next_mention_latents(
                state, writer_lang_hs, lens, mention_beliefs,
                mention_latent_beliefs,
                dots_mentioned_num_markables_to_force=next_dots_mentioned_num_markables if force_next_mention_num_markables else None,
                can_confirm=can_confirm_next,
            )
            # next_mention_out = self.next_mention_prediction(
            #     state, writer_lang_hs, lens, mention_beliefs,
            #     num_markables_to_force=next_num_markables if force_next_mention_num_markables else None,
            # )
            # assert next_mention_out is not None
        else:
            next_mention_latents = None

        if compute_sel_out:
            # compute selection
            # print('sel_idx size: {}'.format(sel_idx.size()))
            selection_beliefs = state.make_beliefs(
                'selection', timestep, partner_ref_outs + [partner_ref_out], ref_outs + [ref_out]
            )
            sel_out = self.selection(state, reader_lang_hs, sel_idx, beliefs=selection_beliefs)
        else:
            sel_out = None

        state = state._replace(turn=state.turn+1)

        return state, outs, (ref_out, partner_ref_out), sel_out, ctx_attn_prob, feed_ctx_attn_prob, next_mention_latents

    def forward(self, ctx, inpt, ref_inpt, sel_idx, num_markables, partner_num_markables, lens,
                dots_mentioned, dots_mentioned_per_ref, dots_mentioned_num_markables,
                belief_constructor: Union[BeliefConstructor, None], partner_ref_inpt, compute_l1_scores=False,
                ref_tgt=None, partner_ref_tgt=None, return_all_selection_outs=False):
        raise NotImplementedError("make this use _State, confirm, can_confirm")
        # belief_function:
        # timestep 0
        if belief_constructor is not None:
            selection_beliefs = belief_constructor.make_beliefs("selection_beliefs", 0, [], [])
            generation_beliefs = belief_constructor.make_beliefs("generation_beliefs", 0, [], [])
            mention_beliefs = belief_constructor.make_beliefs("mention_beliefs", 0, [], [])
            if selection_beliefs is not None:
                raise NotImplementedError("selection_belief for non-hierarchical model")
            if generation_beliefs is not None:
                raise NotImplementedError("selection_belief for non-hierarchical model")
            if mention_beliefs is not None:
                raise NotImplementedError("mention_belief for non-hierarchical model")

        ctx_h = self.ctx_encoder(ctx)
        ctx_differences = self.ctx_differences(ctx)

        reader_and_writer_lang_h = None
        bsz = ctx.size(0)
        dot_h = self._init_dot_h_maybe_multi(bsz, False)
        dot_h_structured = self._init_dot_h_maybe_multi(bsz, True)

        is_selection_out = None

        outs, (ref_out, partner_ref_out), sel_out, reader_and_writer_lang_h, ctx_attn_prob, feed_ctx_attn_prob, next_mention_out, dot_h = self._forward(
            ctx, ctx_differences, ctx_h, inpt, ref_inpt, sel_idx,
            num_markables, partner_num_markables,
            lens=lens, reader_and_writer_lang_h=reader_and_writer_lang_h, compute_sel_out=True,
            dots_mentioned=dots_mentioned, dots_mentioned_per_ref=dots_mentioned_per_ref,
            belief_constructor=belief_constructor,
            partner_ref_inpt=partner_ref_inpt, timestep=0, partner_ref_outs=[],
            dot_h=dot_h, ref_tgt=ref_tgt, partner_ref_tgt=partner_ref_tgt,
        )
        return outs, (ref_out, partner_ref_out), sel_out, ctx_attn_prob, feed_ctx_attn_prob, next_mention_out, is_selection_out, reader_and_writer_lang_h, ctx_h, ctx_differences

    def _context_for_feeding(self, ctx_differences, ctx_h, lang_hs, dots_mentioned, dots_mentioned_per_ref,
                             generation_beliefs, structured_generation_beliefs, ctx):
        if self.args.feed_context_attend:
            # bsz x num_dots
            feed_ctx_attn_prob = self._language_conditioned_dot_attention(
                ctx_differences, ctx_h, lang_hs,
                attention_type='feed',
                dots_mentioned=dots_mentioned,
                dots_mentioned_per_ref=dots_mentioned_per_ref,
                generation_beliefs=generation_beliefs,
                structured_generation_beliefs=structured_generation_beliefs,
                ctx=ctx,
            ).squeeze(0).squeeze(-1)
            # bsz x num_dots x nembed_ctx
            ctx_emb = self.feed_ctx_layer(ctx_h)
            ctx_emb = torch.einsum("bn,bnd->bd", (feed_ctx_attn_prob, ctx_emb))
        else:
            # bsz x nembed_ctx
            ctx_emb = self.feed_ctx_layer(ctx_h).mean(1)
            feed_ctx_attn_prob = None
        return ctx_emb, feed_ctx_attn_prob

    def _context_for_hidden(self, state: State, lang_hs, dots_mentioned, dots_mentioned_per_ref,
                            dots_mentioned_num_markables, generation_beliefs, structured_generation_beliefs):
        ctx = state.ctx
        ctx_h = state.ctx_h
        ctx_differences = state.ctx_differences
        device = ctx_h.device
        # bsz x num_dots
        if self.args.hidden_context_mention_encoder:
            bsz = ctx_h.size(0)
            ctx_emb = self.hidden_ctx_encoder_no_markables.unsqueeze(0).repeat_interleave(bsz, dim=0)

            if dots_mentioned_per_ref is None:
                return ctx_emb, None, None, None

            bsz_, max_num_markables, _ = dots_mentioned_per_ref.size()
            assert bsz == bsz_
            assert max_num_markables == dots_mentioned_num_markables.max().item()
            # bsz x hidden_dim
            # now for utterances that have markables, summarize them

            encoded = torch.zeros(bsz, max_num_markables, self.args.nembed_ctx).to(device)
            encoded_masks = torch.zeros(bsz, max_num_markables).bool().to(device)

            if dots_mentioned_num_markables.max() > 0:
                # bsz x max_num_mentions x num_dots
                dmpr_filtered = dots_mentioned_per_ref[dots_mentioned_num_markables > 0]

                # bsz x num_dots
                dm_filtered = dmpr_filtered.max(1).values.float()
                num_markables_filtered = dots_mentioned_num_markables[dots_mentioned_num_markables > 0]

                encoder_type = vars(self.args).get('hidden_context_mention_encoder_type', 'full')
                if encoder_type == 'filtered-shared':
                    ctx_h_filtered = self.ctx_encoder(ctx[dots_mentioned_num_markables > 0],
                                                      relational_dot_mask=dm_filtered)
                elif encoder_type == 'filtered-separate':
                    ctx_h_filtered = self.hidden_ctx_encoder_relational(ctx[dots_mentioned_num_markables > 0],
                                                                        relational_dot_mask=dm_filtered)
                elif encoder_type == 'full':
                    ctx_h_filtered = ctx_h[dots_mentioned_num_markables > 0]
                else:
                    raise ValueError(f"invalid --hidden_context_mention_encoder_type={encoder_type}")

                # select the embeddings for the dots mentioned in each utterance
                # filtered_bsz x max_num_markables x num_dots x ctx_dim
                to_select = ctx_h_filtered
                if vars(self.args).get('hidden_context_mention_encoder_dot_recurrence', False):
                    if self.args.dot_recurrence_split:
                        assert isinstance(state.dot_h_maybe_multi, tuple)
                        dot_h = torch.cat(state.dot_h_maybe_multi, -1)
                    else:
                        dot_h = state.dot_h_maybe_multi
                    dot_h = dot_h[dots_mentioned_num_markables > 0]
                    assert dot_h.size(0) == dmpr_filtered.size(0)
                    to_select = torch.cat((to_select, dot_h), -1)

                # to_select: filtered_bsz x 7 x dim
                # dmpr_filtered: filtered_bsz x max_num_markables x 7
                ctx_h_selected = to_select.unsqueeze(1).repeat_interleave(max_num_markables, dim=1) * dmpr_filtered.unsqueeze(-1)
                ctx_h_selected = einops.reduce(ctx_h_selected, "b mnm nd c -> b mnm c", 'sum')

                # filtered_bsz x max_num_mentions
                # take the mean over all mention dots, for those with non-zero dots mentioned
                num_dots_mentioned_per_ref = einops.reduce(dmpr_filtered, "b mnm nd -> b mnm", 'sum')
                # clamp to prevent dividing by zero
                ctx_h_selected /= torch.clamp(num_dots_mentioned_per_ref.unsqueeze(-1), min=1.0)
                # use self.hidden_ctx_encoder_no_dots for any others: use this outer-product hack because we can't masked_fill with a vector-valued fill
                ctx_h_selected += torch.einsum("bm,h->bmh", ((num_dots_mentioned_per_ref == 0).float(), self.hidden_ctx_encoder_no_dots))
                to_encode = ctx_h_selected

                if vars(self.args).get('hidden_context_mention_encoder_count_features', False):
                    to_encode = torch.cat((to_encode, num_dots_mentioned_per_ref.float().unsqueeze(-1)), -1)

                if vars(self.args).get('hidden_context_mention_encoder_property_diffs', False):
                    ctx_filtered = ctx[dots_mentioned_num_markables > 0].view(-1, self.num_ent, 4)
                    ctx_selected = ctx_filtered.unsqueeze(1).repeat_interleave(max_num_markables, dim=1) * dmpr_filtered.unsqueeze(-1)
                    ctx_selected = einops.reduce(ctx_selected, "b mnm nd c -> b mnm c", 'sum')
                    ctx_selected /= torch.clamp(num_dots_mentioned_per_ref.unsqueeze(-1), min=1.0)
                    # use self.hidden_ctx_encoder_no_dots_prop_diff for any others: use this outer-product hack because we can't masked_fill with a vector-valued fill
                    ctx_selected += torch.einsum("bm,h->bmh", ((num_dots_mentioned_per_ref == 0).float(), self.hidden_ctx_encoder_no_dots_prop_diff))

                if self.args.hidden_context_mention_encoder_diffs:
                    # filtered_bsz x max_num_mentions x nembed_ctx
                    dots_selected_pad = ctx_h_selected.clone()
                    if vars(self.args).get('hidden_context_mention_encoder_property_diffs', False):
                        dots_selected_pad = torch.cat(
                            (dots_selected_pad, ctx_selected), -1
                        )
                    dots_selected_pad = torch.cat(
                        (dots_selected_pad, torch.zeros((dots_selected_pad.size(0), 1, dots_selected_pad.size(-1)))),
                        1)
                    for bix, nm in enumerate(num_markables_filtered):
                        dots_selected_pad[bix,nm:] = self.hidden_ctx_encoder_diff_pad.unsqueeze(0)
                    dots_selected_diffs = dots_selected_pad[:,:-1] - dots_selected_pad[:,1:]
                    to_encode = torch.cat((to_encode, dots_selected_diffs), -1)

                with set_temporary_default_tensor_type(torch.FloatTensor):
                    # need this since the default tensor type is cuda, and pack_padded_sequence does things on cpu
                    to_encode = pack_padded_sequence(to_encode, num_markables_filtered.cpu().long(),
                                                     batch_first=True, enforce_sorted=False)
                # h: 1 x filtered_bsz x hidden_dim
                encoded_multi, (h, c) = self.hidden_ctx_encoder(to_encode)
                if vars(self.args).get('hidden_context_mention_encoder_bidirectional', False):
                    h = einops.rearrange(h, "dir bsz hidden -> bsz (dir hidden)")
                else:
                    h = h.squeeze(0)
                ctx_emb[dots_mentioned_num_markables > 0] = h.squeeze(0)
                encoded_hs_multi, encoded_lens_multi = pad_packed_sequence(encoded_multi, batch_first=True)
                encoded[dots_mentioned_num_markables > 0] = encoded_hs_multi
                encoded_masks[dots_mentioned_num_markables > 0] = lengths_to_mask(max_num_markables, encoded_lens_multi).to(encoded_masks.device)
            return ctx_emb, None, encoded, encoded_masks
        else:
            ctx_attn_prob = self._language_conditioned_dot_attention(
                ctx_differences, ctx_h, lang_hs,
                attention_type='hidden',
                dots_mentioned=dots_mentioned,
                dots_mentioned_per_ref=dots_mentioned_per_ref,
                generation_beliefs=generation_beliefs,
                structured_generation_beliefs=structured_generation_beliefs,
                ctx=state.ctx,
            ).squeeze(0).squeeze(-1)
            # bsz x num_dots x nembed_ctx
            ctx_emb = self.hidden_ctx_layer(ctx_h)
            ctx_emb = torch.einsum("bn,bnd->bd", (ctx_attn_prob, ctx_emb))
            return ctx_emb, ctx_attn_prob, None, None

    def _init_h(self, bsz):
        if hasattr(self, 'reader_init_h'):
            if vars(self.args).get('bidirectional_reader', False):
                reader_lang_h = torch.stack((self.reader_init_h, self.reader_init_h_reverse), 0)
                reader_lang_h = reader_lang_h.unsqueeze(1).expand(2, bsz, -1).contiguous()
            else:
                reader_lang_h = self.reader_init_h.unsqueeze(0).unsqueeze(1).expand(1, bsz, -1).contiguous()
            writer_lang_h = self.writer_init_h.unsqueeze(0).unsqueeze(1).expand(1, bsz, -1).contiguous()
        else:
            # for backward compatibility, when we didn't have untied grus. TODO: just remove this
            reader_lang_h = self.init_h.unsqueeze(0).unsqueeze(1).expand(1, bsz, -1).contiguous()
            writer_lang_h = reader_lang_h
        return reader_lang_h, writer_lang_h

    def _add_hidden_context(
        self, state: State, reader_lang_h, writer_lang_h, dots_mentioned, dots_mentioned_per_ref,
        dots_mentioned_num_markables, generation_beliefs, structured_generation_beliefs, is_selection,
        can_confirm,
    ):
        if vars(self.args).get('hidden_context', False):
            ctx_emb_for_hidden, _, ctx_seq_encoded, ctx_seq_mask = self._context_for_hidden(
                state, writer_lang_h,
                dots_mentioned=dots_mentioned, dots_mentioned_per_ref=dots_mentioned_per_ref,
                dots_mentioned_num_markables=dots_mentioned_num_markables,
                generation_beliefs=generation_beliefs,
                structured_generation_beliefs=structured_generation_beliefs,
            )
            gate_weights = self.hidden_ctx_gate(
                # TODO: modify this if more than 1 layer
                # reader_lang_h: 1 x bsz x hidden_dim
                torch.cat((writer_lang_h.squeeze(0), ctx_emb_for_hidden), -1)
            )
            addition = self.hidden_ctx_addition(ctx_emb_for_hidden)
            if not self.args.untie_grus:
                assert torch.allclose(reader_lang_h, writer_lang_h)
            writer_lang_h = gate_weights * writer_lang_h + (1 - gate_weights) * addition
            assert self.args.untie_grus
            if vars(self.args).get('hidden_context_is_selection', False):
                assert is_selection is not None
                sig_weight = self.hidden_ctx_is_selection_weight.sigmoid()
                # 1 x bsz x nhid_lang
                is_sel_embs = self.hidden_ctx_is_selection_embeddings(is_selection.long()).unsqueeze(0)
                writer_lang_h = (1-sig_weight) * writer_lang_h + sig_weight * is_sel_embs
            if vars(self.args).get('hidden_context_confirmations', False):
                assert can_confirm is not None
                sig_weight = self.hidden_ctx_confirmations_weight.sigmoid()
                conf_embs = self.hidden_ctx_confirmations_embeddings(can_confirm.long()).unsqueeze(0)
                writer_lang_h = (1-sig_weight) * writer_lang_h + sig_weight * conf_embs
            # if not self.args.untie_grus:
            #     reader_lang_h = writer_lang_h
        else:
            ctx_seq_encoded = None
            ctx_seq_mask = None
        return writer_lang_h, (ctx_seq_encoded, ctx_seq_mask)

    def _attend_encoded_context(self, writer_lang_h, ctx_seq_encoded, ctx_seq_mask):
        pass

    # -> (reader_lang_hs, writer_lang_hs), state, feed_ctx_attn_prob
    def _read(self, state: _State, inpt, lens=None, pack=False, dots_mentioned=None, dots_mentioned_per_ref=None,
              dots_mentioned_num_markables=None, generation_beliefs=None, is_selection=None, can_confirm=None):
        # lang_h: num_layers * num_directions x batch x nhid_lang
        ctx = state.ctx
        ctx_h = state.ctx_h
        ctx_differences = state.ctx_differences
        bsz = ctx_h.size(0)
        seq_len = inpt.size(0)
        num_dots = ctx_h.size(1)

        untie_grus = vars(self.args).get('untie_grus', False)

        reader_lang_h, writer_lang_h = state.reader_and_writer_lang_h

        writer_lang_h, ctx_seq_encoded_and_mask = self._add_hidden_context(
            state, reader_lang_h, writer_lang_h, dots_mentioned, dots_mentioned_per_ref,
            dots_mentioned_num_markables, generation_beliefs, state.dot_h_maybe_multi_structured,
            is_selection, can_confirm,
        )

        # seq_len x batch_size x nembed_word
        dialog_emb = self.embed_dialogue(inpt)
        if self.args.feed_context:
            raise NotImplementedError()
            # seq_len x bsz x (nembed_word+nembed_ctx)
            ctx_emb, feed_ctx_attn_prob = self._context_for_feeding(
                ctx_differences, ctx_h, writer_lang_h,
                dots_mentioned=dots_mentioned, dots_mentioned_per_ref=dots_mentioned_per_ref,
                generation_beliefs=generation_beliefs, structured_generation_beliefs=state.dot_h_maybe_multi_structured,
                ctx=state.ctx,
            )
            dialog_emb = torch.cat((dialog_emb, ctx_emb.expand(seq_len, -1, -1)), dim=-1)
        else:
            feed_ctx_attn_prob = None
        if pack:
            assert lens is not None
            with set_temporary_default_tensor_type(torch.FloatTensor):
                dialog_emb = pack_padded_sequence(dialog_emb, lens.cpu(), enforce_sorted=False)
        # print('lang_h size: {}'.format(lang_h.size()))
        reader_lang_hs, reader_last_h = self.reader(dialog_emb, reader_lang_h)
        if pack:
            reader_lang_hs, _ = pad_packed_sequence(reader_lang_hs, total_length=seq_len)
        if untie_grus:
            writer_lang_hs, writer_last_h = self.writer(dialog_emb, writer_lang_h)
            if pack:
                writer_lang_hs, _ = pad_packed_sequence(writer_lang_hs, total_length=seq_len)
        else:
            writer_lang_hs, writer_last_h = self.reader_forward_hs(reader_lang_hs), self.reader_forward_last_h(reader_last_h)

        state = state._replace(reader_and_writer_lang_h=(reader_last_h, writer_last_h))
        return (reader_lang_hs, writer_lang_hs), state, feed_ctx_attn_prob, ctx_seq_encoded_and_mask

    # a wrapper for _read that makes sure 'THEM:' or 'YOU:' is at the beginning
    # -> (reader_lang_hs, writer_lang_hs), state,
    def read(self, state: State, inpt, prefix_token='THEM:',
             dots_mentioned=None, dots_mentioned_per_ref=None,
             dots_mentioned_num_markables=None,
             generation_beliefs=None, is_selection=None, can_confirm=None):
        # Add a 'THEM:' token to the start of the message
        prefix = self.word2var(prefix_token).unsqueeze(0)
        inpt = torch.cat([prefix, inpt])
        reader_and_writer_lang_hs, state, feed_ctx_attn_prob, ctx_seq_encoded_and_mask = self._read(
            state, inpt, lens=None, pack=False,
            dots_mentioned=dots_mentioned,
            dots_mentioned_per_ref=dots_mentioned_per_ref,
            dots_mentioned_num_markables=dots_mentioned_num_markables,
            generation_beliefs=generation_beliefs,
            is_selection=is_selection,
            can_confirm=can_confirm,
        )
        return reader_and_writer_lang_hs, state

    def word2var(self, word):
        x = torch.Tensor(1).fill_(self.word_dict.get_idx(word)).long()
        return Variable(x)

    def words2var(self, words):
        return torch.Tensor([self.word_dict.get_idx(word) for word in words]).long()

    def write(self, state: State, max_words, temperature,
              start_token='YOU:', stop_tokens=data.STOP_TOKENS, force_words=None,
              dots_mentioned=None, dots_mentioned_per_ref=None, dots_mentioned_num_markables=None,
              generation_beliefs=None, is_selection=None, can_confirm=None):
        # ctx_h: batch x num_dots x nembed_ctx
        # lang_h: batch x hidden
        ctx = state.ctx
        ctx_h = state.ctx_h
        ctx_differences = state.ctx_differences
        reader_lang_h, writer_lang_h = state.reader_and_writer_lang_h
        num_directions, bsz, _ = reader_lang_h.size()
        bsz_, num_dots, _ = ctx_h.size()
        assert bsz == bsz_
        # autoregress starting from start_token
        inpt = self.word2var(start_token)

        assert bsz == 1

        if force_words is not None:
            assert isinstance(force_words, list) and len(force_words) == bsz
            force_words = [
                x[1:] if x[0] == start_token else x
                for x in force_words
            ]

        outs = []
        logprobs = []
        lang_hs = []

        writer_lang_h, ctx_seq_encoded_and_mask = self._add_hidden_context(
            state, reader_lang_h, writer_lang_h, dots_mentioned, dots_mentioned_per_ref,
            dots_mentioned_num_markables, generation_beliefs, state.dot_h_maybe_multi_structured,
            is_selection, can_confirm,
        )

        if self.args.feed_context:
            ctx_emb, feed_ctx_attn_prob = self._context_for_feeding(
                ctx_differences, ctx_h, writer_lang_h, dots_mentioned, dots_mentioned_per_ref,
                generation_beliefs, state.dot_h_maybe_multi_structured, ctx=state.ctx,
            )
        else:
            ctx_emb, feed_ctx_attn_prob = None, None

        ctx_attn_probs = []
        top_words = []
        top_indices = []
        for word_ix in range(max_words):
            # embed
            inpt_emb = self.embed_dialogue(inpt)
            if ctx_emb is not None:
                # bsz x (nembed_word+nembed_ctx)
                inpt_emb = torch.cat((inpt_emb, ctx_emb), dim=-1)

            # we'll use dim 0 for both num_directions (on initialization) and time (in lang_hs, and the _language_conditioned_dot_attention, etc methods)
            # bsz x hidden_dim
            writer_lang_h = self.writer_cell(inpt_emb, writer_lang_h.squeeze(0)).unsqueeze(0)
            lang_hs.append(writer_lang_h)

            if self.word_dict.get_word(inpt.data[0]) in stop_tokens:
                break

            out, ctx_attn_prob = self.language_output(
                state, writer_lang_h, dots_mentioned, dots_mentioned_per_ref, generation_beliefs,
                ctx_seq_encoded_and_mask=ctx_seq_encoded_and_mask
            )
            # remove time dimension
            out = out.squeeze(0)

            # TODO: check whether we need to squeeze (or unsqueeze) this
            ctx_attn_probs.append(ctx_attn_prob)

            logprob_no_temp = F.log_softmax(out, dim=-1)

            scores = out.div(temperature)
            scores = scores.sub(scores.max().item())

            mask = Variable(self.special_token_mask.to(scores.device))
            scores = scores.add(mask.unsqueeze(0))

            prob = F.softmax(scores, dim=-1)
            logprob = F.log_softmax(scores, dim=-1)

            this_top_indices = logprob_no_temp.topk(5, dim=-1)
            top_indices.append(this_top_indices)
            top_words.append(list(zip(self.word_dict.i2w(this_top_indices.indices.flatten().cpu()), this_top_indices.values.flatten().cpu())))

            if force_words is not None:
                inpt = []
                for words in force_words:
                    if word_ix < len(words):
                        inpt.append(self.word2var(words[word_ix]))
                    else:
                        inpt.append(self.word2var('<pad>'))
                inpt = torch.cat(inpt, dim=0)
            else:
                # squeeze out sample dimension
                # batch
                inpt = prob.multinomial(1).squeeze(1).detach()
            outs.append(inpt.unsqueeze(0))

            # print('inpt: {}'.format(inpt.size()))
            # print('logprob: {}'.format(logprob.size()))
            logprob = logprob.gather(-1, inpt.unsqueeze(-1)).squeeze(-1)
            logprobs.append(logprob)

        outs = torch.cat(outs, 0)

        # TODO: consider swapping in partner context

        # read the output utterance
        (reader_lang_hs, writer_lang_hs), state_new = self.read(
            state, outs,
            dots_mentioned=dots_mentioned,
            dots_mentioned_per_ref=dots_mentioned_per_ref,
            dots_mentioned_num_markables=dots_mentioned_num_markables,
            prefix_token=start_token,
            is_selection=is_selection,
            can_confirm=can_confirm,
        )

        assert torch.allclose(writer_lang_hs, torch.cat(lang_hs, 0), atol=1e-4)

        extra = {
            'feed_ctx_attn_prob': feed_ctx_attn_prob,
            'word_ctx_attn_probs': ctx_attn_probs,
            'top_words': top_words,
            'top_indices': top_indices,
        }
        if ctx_attn_probs:
            extra['word_ctx_attn_prob_mean'] = torch.mean(torch.stack(ctx_attn_probs, 0), 0)

        # lang_hs: list of tensors [1 x bsz x hidden_dim, ...]
        return outs, logprobs, state_new, (reader_lang_hs, writer_lang_hs), extra

    def write_beam(self, state, max_words, beam_size,
                   start_token='YOU:', stop_tokens=data.STOP_TOKENS,
                   dots_mentioned=None, dots_mentioned_per_ref=None, dots_mentioned_num_markables=None,
                   generation_beliefs=None, gumbel_noise=False, temperature=1.0,
                   is_selection=None, read_one_best=True, gumbel_noise_forgetful=False, keep_all_finished=False,
                   can_confirm=None,
                   ):
        # NOTE: gumbel_noise=True *does not* sample without replacement at the sequence level, for the reasons
        # outlined in https://arxiv.org/pdf/1903.06059.pdf, and so should just be viewed as a randomized beam search

        # ctx_h: batch x num_dots x nembed_ctx
        # lang_h: batch x hidden
        ctx = state.ctx
        device = ctx.device
        ctx_h = state.ctx_h
        ctx_differences = state.ctx_differences
        reader_lang_h, writer_lang_h = state.reader_and_writer_lang_h
        num_directions, bsz, _ = reader_lang_h.size()
        bsz_, num_dots, _ = ctx_h.size()
        assert bsz == bsz_
        # autoregress starting from start_token

        assert bsz == 1

        writer_lang_h, (ctx_seq_encoded, ctx_seq_mask) = self._add_hidden_context(
            state, reader_lang_h, writer_lang_h, dots_mentioned, dots_mentioned_per_ref,
            dots_mentioned_num_markables, generation_beliefs, state.dot_h_maybe_multi_structured,
            is_selection, can_confirm,
        )

        if self.args.feed_context:
            ctx_emb, feed_ctx_attn_prob = self._context_for_feeding(
                ctx_differences, ctx_h, writer_lang_h, dots_mentioned, dots_mentioned_per_ref,
                generation_beliefs, state.dot_h_maybe_multi_structured, ctx=state.ctx
            )
        else:
            ctx_emb, feed_ctx_attn_prob = None, None

        writer_lang_h_expanded = writer_lang_h.expand(1, beam_size, -1)
        ctx_expanded = ctx.expand(beam_size, -1)
        ctx_h_expanded = ctx_h.expand(beam_size, -1, -1)
        ctx_differences_expanded = ctx_differences.expand(beam_size, -1, -1)
        dots_mentioned_expanded = dots_mentioned.expand(beam_size, -1)
        dots_mentioned_per_ref_expanded = dots_mentioned_per_ref.expand(beam_size, -1, -1)

        if ctx_seq_encoded is not None and ctx_seq_mask is not None:
            ctx_seq_encoded_and_mask_expanded = (
                ctx_seq_encoded.expand(beam_size, -1, -1),
                ctx_seq_mask.expand(beam_size, -1)
            )
        else:
            ctx_seq_encoded_and_mask_expanded = None

        # pass None for vars that really should be expanded, but not implemented for now
        state_expanded = state._replace(
            bsz=beam_size, ctx=ctx_expanded,
            ctx_h=ctx_h_expanded,
            ctx_differences=ctx_differences_expanded,
            reader_and_writer_lang_h=(None, writer_lang_h_expanded),
            dot_h_maybe_multi=None,
            dot_h_maybe_multi_structured=None,
        )

        if generation_beliefs is not None:
            raise NotImplementedError("todo: check the size of generation_beliefs and expand")
        if state.dot_h_maybe_multi_structured is not None:
            raise NotImplementedError("todo: check the size of generation_beliefs and expand")

        outputs = self.word2var(start_token).view(1,1).expand(beam_size, 1)
        is_finished = torch.zeros(beam_size).bool().to(device)
        # prevent the beam from filling with identical copies of the same hypothesis, by breaking ties
        total_scores = torch.full((beam_size,), BIG_NEG).float().to(device)
        total_scores[0] = 0

        total_log_probs = torch.zeros((beam_size,)).float().to(device)
        lens = torch.zeros(beam_size).long().to(device)

        pad_ix = self.word_dict.get_idx('<pad>')

        force_pad_mask = -make_mask(len(self.word_dict), [pad_ix])

        for word_ix in range(max_words):
            if is_finished.all():
                break
            # embed
            # bsz x hidden
            inpt_emb = self.embed_dialogue(outputs[:,-1])
            if ctx_emb is not None:
                # bsz x (nembed_word+nembed_ctx)
                inpt_emb = torch.cat((inpt_emb, ctx_emb), dim=-1)

            this_beam_size = inpt_emb.size(0)

            # we'll use dim 0 for both num_directions (on initialization) and time (in lang_hs, and the _language_conditioned_dot_attention, etc methods)
            # this_beam_size x hidden_dim
            writer_lang_h_expanded = self.writer_cell(inpt_emb, writer_lang_h_expanded.squeeze(0)).unsqueeze(0)

            out, _ = self.language_output(
                state_expanded, writer_lang_h_expanded, dots_mentioned_expanded, dots_mentioned_per_ref_expanded,
                generation_beliefs, ctx_seq_encoded_and_mask=ctx_seq_encoded_and_mask_expanded
            )
            # remove time dimension
            out = out.squeeze(0)

            mask = self.special_token_mask.to(out.device)
            word_scores = out.add(mask.unsqueeze(0))

            word_scores = torch.where(is_finished.unsqueeze(1), force_pad_mask.unsqueeze(0), word_scores)

            # this_beam_size x vocab
            word_logprobs = word_scores.log_softmax(dim=-1)
            if temperature != 1.0:
                word_logprobs_to_sample = (word_logprobs / temperature).log_softmax(-1)
            else:
                word_logprobs_to_sample = word_logprobs

            # (this_beam_size * vocab)
            if gumbel_noise_forgetful:
                assert not gumbel_noise
                candidate_scores = Gumbel(word_logprobs_to_sample, scale=1.0).sample()
            elif gumbel_noise:
                candidate_scores = total_scores.unsqueeze(1) + Gumbel(word_logprobs_to_sample, scale=1.0).sample()
            else:
                candidate_scores = total_scores.unsqueeze(1) + word_logprobs_to_sample

            vocab_size = word_logprobs.size(-1)

            top_scores, top_indices = candidate_scores.view(-1).topk(beam_size, dim=-1)
            top_beam_indices = top_indices // vocab_size
            top_vocab_indices = top_indices % vocab_size

            if keep_all_finished:
                num_top_to_keep = beam_size - is_finished.sum()
                finished_beam_indices = is_finished.nonzero().flatten()
                finished_vocab_indices = torch.full(finished_beam_indices.size(), pad_ix).long().to(device)
                finished_beam_indices_set = set(finished_beam_indices.tolist())
                finished_scores = total_scores[is_finished]

                finished_total_log_probs = total_log_probs[is_finished]

                top_beam_indices_to_keep = []
                top_vocab_indices_to_keep = []
                top_scores_to_keep = []

                top_total_log_probs_to_keep = []

                for beam_ix, vocab_ix, score in zip(top_beam_indices, top_vocab_indices, top_scores):
                    if len(top_beam_indices_to_keep) >= num_top_to_keep:
                        break
                    if beam_ix.item() not in finished_beam_indices_set:
                        top_beam_indices_to_keep.append(beam_ix)
                        top_vocab_indices_to_keep.append(vocab_ix)
                        top_scores_to_keep.append(score)
                        top_total_log_probs_to_keep.append(total_log_probs[beam_ix] + word_logprobs[beam_ix,vocab_ix])

                top_beam_indices_to_keep = torch.stack(top_beam_indices_to_keep, -1)
                top_vocab_indices_to_keep = torch.stack(top_vocab_indices_to_keep, -1)
                top_scores_to_keep = torch.stack(top_scores_to_keep, -1)
                top_total_log_probs_to_keep = torch.stack(top_total_log_probs_to_keep, -1)

                top_beam_indices = torch.cat((finished_beam_indices, top_beam_indices_to_keep), -1)
                top_vocab_indices = torch.cat((finished_vocab_indices, top_vocab_indices_to_keep), -1)
                total_scores = torch.cat((finished_scores, top_scores_to_keep), -1)
                total_log_probs = torch.cat((finished_total_log_probs, top_total_log_probs_to_keep), -1)

            else:
                total_scores = top_scores
                total_log_probs = (total_log_probs.unsqueeze(1) + word_logprobs).view(-1).gather(-1, top_indices)

            outputs = outputs[top_beam_indices]
            lens = lens[top_beam_indices]
            is_finished = is_finished[top_beam_indices]
            writer_lang_h_expanded = writer_lang_h_expanded[:,top_beam_indices]

            outputs = torch.cat((outputs, top_vocab_indices.unsqueeze(-1)), -1)
            lens[~is_finished] += 1
            is_finished |= torch.BoolTensor(
                [self.word_dict.get_word(ix.item()) in stop_tokens for ix in top_vocab_indices],
            ).to(device)

        sort_indices = total_scores.argsort(descending=True)
        outputs = outputs[sort_indices]
        lens = lens[sort_indices]
        total_scores = total_scores[sort_indices]
        total_log_probs = total_log_probs[sort_indices]

        # remove start token for consistency with write()
        outputs = outputs[:,1:]
        # this would remove <eos>
        # lens -= 1

        decoded = [
            [self.word_dict.get_word(ix.item()) for ix in sent[:l]]
            for sent, l in zip(outputs, lens)
        ]

        # T x bsz
        best_output = outputs[0][:lens[0]].unsqueeze(1)

        # read the output utterance
        if read_one_best:
            (reader_lang_hs, writer_lang_hs), state_new = self.read(
                state, best_output,
                dots_mentioned=dots_mentioned,
                dots_mentioned_per_ref=dots_mentioned_per_ref,
                dots_mentioned_num_markables=dots_mentioned_num_markables,
                prefix_token=start_token,
                is_selection=is_selection,
                can_confirm=can_confirm,
            )
        else:
            reader_lang_hs = writer_lang_hs = state_new = None

        # if not (gumbel_noise or gumbel_noise_forgetful):
        #     assert torch.allclose(total_scores, total_log_probs)

        extra = {
            'outputs': outputs,
            'lens': lens,
            'words': decoded,
            'output_scores': total_scores,
            'output_logprobs': total_log_probs,
        }

        # None: for where write() returns logprobs
        return best_output, None, state_new, (reader_lang_hs, writer_lang_hs), extra

class HierarchicalRnnReferenceModel(RnnReferenceModel):
    corpus_ty = corpora.reference_sentence.ReferenceSentenceCorpus
    engine_ty = HierarchicalRnnReferenceEngine

    @classmethod
    def add_args(cls, parser):
        # args from RnnReferenceModel will be added separately
        pass

    def first_mention_latents(self, state: State, dots_mentioned_num_markables=None, force_next_mention_num_markables=False,
                              min_num_mentions=0, max_num_mentions=12):
        mention_beliefs = state.make_beliefs('mention', -1, [], [])
        mention_latent_beliefs = state.make_beliefs('next_mention_latents', -1, [], [])
        can_confirm = torch.full((state.bsz,), CAN_CONFIRM_VALUES[None], device=state.ctx_h.device).long()
        return self.next_mention_latents(
            state,
            outs_emb=state.reader_and_writer_lang_h[0], # TODO: fix this to use the writer instead
            lens=torch.full((state.bsz,), 1.0, device=state.ctx_h.device).long(),
            mention_beliefs=mention_beliefs,
            mention_latent_beliefs=mention_latent_beliefs,
            dots_mentioned_num_markables_to_force=dots_mentioned_num_markables if force_next_mention_num_markables else None,
            min_num_mentions=min_num_mentions,
            max_num_mentions=max_num_mentions,
            can_confirm=can_confirm,
        )

    def first_mention(self, state: State, dots_mentioned_num_markables=None, force_next_mention_num_markables=False,
                      min_num_mentions=0, max_num_mentions=12):
        latents: NextMentionLatents = self.first_mention_latents(
            state, dots_mentioned_num_markables, force_next_mention_num_markables, min_num_mentions, max_num_mentions
        )
        return self.next_mention_prediction_from_latents(state, latents)

    def current_selection_probabilities(self, state: State):
        # this may be too stringent
        assert 'predicted_hidden' not in self.args.dot_recurrence_inputs and 'weighted_hidden' not in self.args.dot_recurrence_inputs
        beliefs = state.make_beliefs(
            'selection', state.turn, [None] * (state.turn + 1), [None] * (state.turn + 1),
        )
        device = state.ctx.device
        dummy_reader_h = torch.zeros((1, state.bsz, self.args.nhid_lang*self.num_reader_directions)).to(device)
        dummy_sel_idx = torch.zeros(state.bsz).long().to(device)
        return self.selection(
            state, dummy_reader_h, dummy_sel_idx, beliefs
        )[0].softmax(-1)

    def rollout_selection_probabilities(self, state: State, dots_mentioned_per_ref, num_markables):
        num_mentions, bsz, num_dots = dots_mentioned_per_ref.size()
        assert state.bsz == bsz
        assert num_dots == 7
        device = state.ctx.device

        reader_h_dim = self.args.nhid_lang * self.num_reader_directions

        if num_mentions == 0:
            dummy_ref_inpt = None
            ref_out = None
        else:
            dummy_ref_inpt = torch.zeros(bsz, num_mentions, 3).long().to(device)
            ref_out_marginals = models.utils.unsigmoid(dots_mentioned_per_ref)
            ref_out_joint = torch.full((num_mentions, bsz, 2**num_dots), -1e9).to(device)
            ref_out_joint.scatter_(-1, bit_to_int_array(dots_mentioned_per_ref).unsqueeze(-1), 0)
            ref_out_dist = None
            ref_out = ref_out_marginals, ref_out_joint, ref_out_dist
        partner_ref_inpt = None
        partner_num_markables = torch.zeros(state.bsz).long().to(device)

        if self.args.dot_recurrence_oracle:
            ref_tgt = dots_mentioned_per_ref.transpose(0,1)
        else:
            ref_tgt = torch.zeros((bsz, num_mentions, num_dots)).long().to(device)
        dummy_partner_ref_tgt = torch.zeros((bsz, 0, num_dots)).long().to(device)
        dummy_reader_h = torch.zeros((1, bsz, reader_h_dim)).to(device)

        new_state = self._update_dot_h_maybe_multi(
            state, dummy_reader_h,
            dummy_ref_inpt, partner_ref_inpt,
            num_markables, partner_num_markables,
            ref_out, None,
            ref_tgt, dummy_partner_ref_tgt,
        )

        return self.current_selection_probabilities(new_state)

    def rollout_next_mention_cands(self, state: State, next_mention_latents: NextMentionLatents,
                                   num_candidates=None, generation_method='topk', use_stop_losses=False):
        current_sel_probs = self.current_selection_probabilities(state)
        if num_candidates is None:
            return []
        if generation_method in ['topk', 'topk_multi_mention']:
            sample = False
            multi_mention = generation_method == 'topk_multi_mention'
        elif generation_method == 'sample':
            sample = True
            multi_mention = False
        else:
            raise NotImplementedError(f"next_mention_candidates_generation=={generation_method}")
        if (next_mention_latents.dots_mentioned_num_markables == 0).all():
            return NextMentionRollouts(
                current_sel_probs, next_mention_latents.dots_mentioned_num_markables, None, None, None, None
            )

        def filter_latents(next_mention_latents, max_mentions):
            return next_mention_latents._replace(
                dots_mentioned_num_markables=next_mention_latents.dots_mentioned_num_markables.clamp_max(max_mentions),
                latent_states=next_mention_latents.latent_states[:max_mentions],
                stop_losses=next_mention_latents.stop_losses[:max_mentions+1],
                ctx_h_with_beliefs=next_mention_latents.ctx_h_with_beliefs[:max_mentions],
            )

        def process_latents(next_mention_latents):
            # TODO: incorporate stop_losses
            next_mention_outs = self.next_mention_prediction_from_latents(state, next_mention_latents)
            next_mention_out, _, _ = next_mention_outs
            stop_scores = -1 * next_mention_latents.stop_losses.sum(0)
            candidate_indices, candidate_dots, candidate_nm_scores = make_candidates(
                next_mention_out, next_mention_latents.dots_mentioned_num_markables, num_candidates, sample,
                additional_scores=stop_scores if use_stop_losses else None
            )
            for b in range(state.bsz):
                nm_b = next_mention_latents.dots_mentioned_num_markables[b]
                # scores should be tiled across num_mentions
                for j in range(nm_b):
                    assert torch.allclose(candidate_nm_scores[j,b], candidate_nm_scores[0,b])
            candidate_nm_scores = candidate_nm_scores[0]

            rollout_sel_probs = []
            for k in range(candidate_dots.size(2)):
                probs_k = self.rollout_selection_probabilities(state, candidate_dots[:,:,k], next_mention_latents.dots_mentioned_num_markables)
                rollout_sel_probs.append(probs_k)

            # bsz x candidates x num_dots
            rollout_sel_probs = torch.stack(rollout_sel_probs, 1)

            num_markables_per_candidate = next_mention_latents.dots_mentioned_num_markables.unsqueeze(1).expand(-1, num_candidates)
            return num_markables_per_candidate, candidate_indices, candidate_dots, candidate_nm_scores, rollout_sel_probs

        if not multi_mention:
            num_markables_per_candidate, candidate_indices, candidate_dots, candidate_nm_scores, rollout_sel_probs = process_latents(next_mention_latents)
        else:
            all_num_markables_per_candidate, all_candidate_indices, all_candidate_dots, all_candidate_nm_scores, all_rollout_sel_probs = zip(
                *[
                    process_latents(filter_latents(next_mention_latents, max_mentions))
                    for max_mentions in range(1, next_mention_latents.dots_mentioned_num_markables.max() + 1)
                ]
            )
            num_markables_per_candidate = torch.cat(all_num_markables_per_candidate, dim=1)
            candidate_indices = einops.rearrange(
                pad_sequence(all_candidate_indices),
                "stack mentions bsz candidates -> mentions bsz (stack candidates)"
            )
            candidate_dots = einops.rearrange(
                pad_sequence(all_candidate_dots),
                "stack mentions bsz candidates dots -> mentions bsz (stack candidates) dots"
                )
            candidate_nm_scores = torch.cat(all_candidate_nm_scores, dim=1)
            rollout_sel_probs = torch.cat(all_rollout_sel_probs, dim=1)

        return NextMentionRollouts(
            current_sel_probs, num_markables_per_candidate, candidate_indices, candidate_dots, candidate_nm_scores, rollout_sel_probs
        )

    def forward(self, ctx, inpts, ref_inpts, sel_idx, num_markables, partner_num_markables, lens,
                dots_mentioned, dots_mentioned_per_ref, dots_mentioned_num_markables,
                belief_constructor: Union[BeliefConstructor, None],
                partner_ref_inpts,
                compute_l1_probs=False, tgts=None, ref_tgts=None, partner_ref_tgts=None,
                force_next_mention_num_markables=False,
                is_selection=None,
                can_confirm=None,
                num_next_mention_candidates_to_score=None,
                next_mention_candidates_generation='topk',
                return_all_selection_outs=False,
                relation_swap=False,
                ):
        # inpts is a list, one item per sentence
        # ref_inpts also a list, one per sentence
        # sel_idx is index into the last sentence in the dialogue

        state = self.initialize_state(ctx, belief_constructor)

        all_outs = []
        all_ref_outs = []

        sel_outs = []

        all_ctx_attn_prob = []
        all_feed_ctx_attn_prob = []

        all_next_mention_outs = []
        all_next_mention_candidates = []

        all_is_selection_outs = []

        all_reader_lang_h = [None]
        all_writer_lang_h = [None]

        # if generation_beliefs is not None:
        #     assert len(generation_beliefs) == len(inpts)
        #
        # if partner_ref_inpts is not None:
        #     assert len(partner_ref_inpts) == len(inpts)

        partner_ref_outs = []
        ref_outs = []

        relation_swapped_ref_outs = []
        relation_swapped_partner_ref_outs = []
        has_relation_swaps = []

        if vars(self.args).get('next_mention_prediction', False):
            # next_mention_outs = self.first_mention(state, num_markables[0], force_next_mention_num_markables)
            next_mention_latents = self.first_mention_latents(state, dots_mentioned_num_markables[0], force_next_mention_num_markables)
            next_mention_outs = self.next_mention_prediction_from_latents(state, next_mention_latents)
            all_next_mention_outs.append(next_mention_outs)
            if num_next_mention_candidates_to_score is not None:
                all_next_mention_candidates.append(self.rollout_next_mention_cands(
                    state, next_mention_latents, num_candidates=num_next_mention_candidates_to_score,
                    generation_method=next_mention_candidates_generation
                ))
                # all_next_mention_candidates.append(self.rollout_next_mention_cands(
                #     state, next_mention_outs[0], next_mention_outs[2],
                #     num_next_mention_candidates_to_score, next_mention_candidates_generation,
                # ))


        l1_log_probs = []
        assert len(inpts) == len(ref_inpts)

        for i in range(len(inpts)):
            inpt = inpts[i]
            ref_inpt = ref_inpts[i]
            is_last = i == len(inpts) - 1
            this_lens = lens[i]
            assert ref_tgts is not None
            ref_tgt = ref_tgts[i] if ref_tgts is not None else None
            partner_ref_tgt = partner_ref_tgts[i] if partner_ref_tgts is not None else None
            is_selection_out = self.is_selection_prediction(state)
            all_is_selection_outs.append(is_selection_out)
            kwargs = dict(
                lens=this_lens,
                ref_inpt=ref_inpt,
                partner_ref_inpt=partner_ref_inpts[i] if partner_ref_inpts is not None else None,
                num_markables=num_markables[i],
                partner_num_markables=partner_num_markables[i],
                compute_sel_out=is_last or return_all_selection_outs, sel_idx=sel_idx,
                dots_mentioned=dots_mentioned[i] if dots_mentioned is not None else None,
                dots_mentioned_per_ref=dots_mentioned_per_ref[i] if dots_mentioned_per_ref is not None else None,
                dots_mentioned_num_markables=dots_mentioned_num_markables[i],
                timestep=i,
                ref_outs=ref_outs, partner_ref_outs=partner_ref_outs,
                ref_tgt=ref_tgt, partner_ref_tgt=partner_ref_tgt,
                force_next_mention_num_markables=force_next_mention_num_markables,
                next_dots_mentioned_num_markables=dots_mentioned_num_markables[i + 1] if i < len(dots_mentioned_num_markables) - 1 else None,
                is_selection=is_selection[i],
                can_confirm=can_confirm[i],
                can_confirm_next=can_confirm[i+1] if i < len(can_confirm) - 1 else None,
            )
            new_state, outs, ref_out_and_partner_ref_out, sel_out, ctx_attn_prob, feed_ctx_attn_prob, next_mention_latents = self._forward(
                state, inpt, **kwargs,
            )

            if relation_swap:
                assert inpt.dim() == 2
                inpt_replaced = torch.tensor(
                    [[self.spatial_replacements.get(ix.item(), ix.item()) for ix in row] for row in inpt]
                ).long().to(inpt.device)
                # only run the forward pass on those inputs that have had some word replaced
                has_relation_swap = (inpt != inpt_replaced).any(0)
                has_relation_swaps.append(has_relation_swap)

                if has_relation_swap.any():
                    masked_kwargs = {}
                    max_num_markables = kwargs['num_markables'][has_relation_swap].max()
                    max_partner_num_markables = kwargs['partner_num_markables'][has_relation_swap].max()
                    def mask(key, val):
                        if isinstance(val, torch.Tensor):
                            # this only works because all of these inputs have the batch dimension first
                            assert val.size(0) == has_relation_swap.size(0)
                            val = val[has_relation_swap]
                            if key in {'ref_inpt', 'ref_tgt'}:
                                if max_num_markables.item() == 0:
                                    val = None
                                else:
                                    val = val[:,:max_num_markables]
                            elif key in {'partner_ref_inpt', 'partner_ref_tgt'}:
                                if max_partner_num_markables.item() == 0:
                                    val = None
                                else:
                                    val = val[:,:max_partner_num_markables]
                            elif key in {'dots_mentioned_per_ref'}:
                                val = val[:,:max_num_markables]
                        if isinstance(val, list):
                            return [mask(key, v) for v in val]
                        return val

                    # for key, val in kwargs.items():
                    #     masked_kwargs[key] = mask(key, val)
                    #
                    # rs_ref_out, rs_partner_ref_out = self._forward(
                    #     # inputs has the batch dimension second
                    #     state.mask(has_relation_swap), inpt_replaced[:,has_relation_swap], **masked_kwargs
                    # )[2]

                    rs_ref_out, rs_partner_ref_out = self._forward(
                        state, inpt_replaced, **kwargs
                    )[2]
                    relation_swapped_ref_outs.append(rs_ref_out)
                    relation_swapped_partner_ref_outs.append(rs_partner_ref_out)
                else:
                    relation_swapped_ref_outs.append(None)
                    relation_swapped_partner_ref_outs.append(None)

            sel_outs.append(sel_out)

            if self.args.next_mention_prediction:
                next_mention_outs = self.next_mention_prediction_from_latents(state, next_mention_latents)
            else:
                next_mention_outs = None

            state = new_state
            if compute_l1_probs and ref_inpt is not None:
                assert tgts is not None
                tgt = tgts[i]
                # TODO: consider filtering down the inputs here using the mask (but needs filtering belief_constructor parameters too, or running without beliefs)
                # mask = is_self[i]
                # if self.args.max_mentions_in_generation_training is not None:
                #     mask = mask & (num_markables[i] <= self.args.max_mentions_in_generation_training)
                scoring_function = self.make_l1_scoring_function(
                    state, inpt, tgt, ref_inpt,
                    num_markables[i], partner_num_markables[i], this_lens,
                    belief_constructor=belief_constructor,
                    partner_ref_inpt=partner_ref_inpts[i] if partner_ref_inpts is not None else None,
                    timestep=i,
                    partner_ref_outs=partner_ref_outs,
                    ref_outs=ref_outs,
                    next_mention_out=next_mention_outs,
                )

                max_num_mentions = ref_inpt.size(1)
                if self.args.l1_normalizer_sampling == 'none':
                    num_candidates = 2**self.num_ent
                    if self.args.max_mentions_in_generation_training != 1:
                        raise NotImplementedError()
                    candidate_indices = torch.arange(0, num_candidates)
                    candidate_indices = candidate_indices.unsqueeze(0).repeat_interleave(max_num_mentions, dim=0)
                    # max_num_mentions x bsz x 2**7
                    candidate_indices = candidate_indices.unsqueeze(1).repeat_interleave(bsz, dim=1)
                elif self.args.l1_normalizer_sampling in ['noised', 'uniform']:
                    num_candidates = self.args.l1_normalizer_sampling_candidates
                    assert num_candidates is not None
                    if self.args.l1_normalizer_sampling == 'noised':
                        # max_num_mentions x bsz x 7
                        # define sampling probabilities: no dot[0]: 0.1; dot[1]: 0.9
                        ref_tgt_logits_t = (ref_tgt.transpose(0,1) * 0.8 + 0.1).log()
                        # max_num_mentions x bsz x 2 x 2 x ...
                        joint_logits = StructuredAttentionLayer.marginal_logits_to_full_logits(ref_tgt_logits_t).contiguous()
                        # max_num_mentions x bsz x_num_mentions x 2**7
                        joint_logits = joint_logits.view(joint_logits.size(0), joint_logits.size(1), -1)
                    elif self.args.l1_normalizer_sampling == 'uniform':
                        joint_logits = torch.zeros(max_num_mentions, state.bsz, 2**self.num_ent)
                    # max_num_mentions x bsz x 2**7
                    noised_logits = Gumbel(joint_logits, scale=1.0).sample()
                    # sample each mention independently
                    # max_num_mentions x bsz x num_candidates
                    candidate_indices = noised_logits.topk(num_candidates, dim=-1).indices
                elif self.args.l1_normalizer_sampling == 'next_mention':
                    raise NotImplementedError("need a hierarchical next mention model")
                else:
                    raise ValueError(f"invalid --l1_normalizer_sampling={self.args.l1_normalizer_sampling}")

                # TODO: this may be double-counting dot configurations that differ beyond the num_markables for that instance
                ## ensure candidates includes the gold
                # bsz , with entries ranging from 0 to 2**(7*max_num_mentions)
                ref_tgt_indices_joint = bit_to_int_array(ref_tgt.view(ref_tgt.size(0), -1))

                # max_num_mentions x bsz x num_candidates x 2**7
                candidates = int_to_bit_array(candidate_indices, num_bits=self.num_ent)
                # bsz x num_candidates
                candidate_indices_joint = bit_to_int_array(
                    einops.rearrange(candidates, "mnm b n d -> b n (mnm d)")
                )
                is_present = (candidate_indices_joint == ref_tgt_indices_joint.unsqueeze(-1)).any(-1)

                # for any indices that don't contain the gold, replace them with the gold
                candidate_indices_joint[:,0] = torch.where(is_present, candidate_indices_joint[:,0], ref_tgt_indices_joint)
                # check that it's present exactly once
                assert ((candidate_indices_joint == ref_tgt_indices_joint.unsqueeze(-1)).sum(-1) == 1).all()

                # bsz
                gold_candidate_indices = (candidate_indices_joint == ref_tgt_indices_joint.unsqueeze(-1)).float().argmax(-1)

                candidates = einops.rearrange(
                    int_to_bit_array(candidate_indices_joint, num_bits=self.num_ent * max_num_mentions),
                    "b n (mnm d) -> mnm b n d",
                    mnm=max_num_mentions, d=self.num_ent,
                )

                assert candidates.size() == (max_num_mentions, state.bsz, num_candidates, self.num_ent)

                # log p(d | u)
                # bsz x num_candidates
                this_l1_log_probs_all = scoring_function(candidates, normalize_over_candidates=True)

                # bsz
                this_l1_log_probs = this_l1_log_probs_all.gather(-1, gold_candidate_indices.unsqueeze(-1)).squeeze(-1)

                l1_log_probs.append(this_l1_log_probs)
            else:
                l1_log_probs.append(None)

            # print("i: {}\tmention_belief.sum(): {}\t(next_mention_out > 0).sum(): {}".format(i, mention_beliefs.sum(), (next_mention_out > 0).sum()))
            all_outs.append(outs)
            all_ref_outs.append(ref_out_and_partner_ref_out)
            all_ctx_attn_prob.append(ctx_attn_prob)
            all_feed_ctx_attn_prob.append(feed_ctx_attn_prob)
            all_next_mention_outs.append(next_mention_outs)

            if num_next_mention_candidates_to_score is not None:
                # all_next_mention_candidates.append(self.rollout_next_mention_cands(
                #     state, next_mention_outs[0], next_mention_outs[2]
                # ))
                all_next_mention_candidates.append(self.rollout_next_mention_cands(
                    state, next_mention_latents,
                ))

            reader_lang_h, writer_lang_h = state.reader_and_writer_lang_h
            all_reader_lang_h.append(reader_lang_h)
            all_writer_lang_h.append(writer_lang_h)

            ref_out, partner_ref_out = ref_out_and_partner_ref_out
            ref_outs.append(ref_out)
            partner_ref_outs.append(partner_ref_out)

        assert len(sel_outs) > 0
        if return_all_selection_outs:
            sel_to_return = sel_outs
        else:
            sel_to_return = sel_outs[-1]

        return state, all_outs, all_ref_outs, sel_to_return, all_ctx_attn_prob, all_feed_ctx_attn_prob, all_next_mention_outs,\
               all_is_selection_outs, (all_reader_lang_h, all_writer_lang_h), l1_log_probs, all_next_mention_candidates, \
            relation_swapped_ref_outs, relation_swapped_partner_ref_outs, has_relation_swaps

