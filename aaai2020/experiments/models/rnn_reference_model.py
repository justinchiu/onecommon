import torch.nn.init

import corpora.reference
import corpora.reference_sentence
from corpora import data
import models
from domain import get_domain
from engines.rnn_reference_engine import RnnReferenceEngine, HierarchicalRnnReferenceEngine
from models.ctx_encoder import *

from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from utils import set_temporary_default_tensor_type

BIG_NEG = -1e6

class FeedForward(nn.Module):
    def __init__(self, n_hidden_layers, input_dim, hidden_dim, output_dim, dropout_p=None):
        super(FeedForward, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.dropout_p = dropout_p
        self.hidden_dim = hidden_dim
        layers = []
        for ix in range(n_hidden_layers + 1):
            this_in = input_dim if ix == 0 else hidden_dim
            is_last = ix == n_hidden_layers
            this_out = output_dim if is_last else hidden_dim
            layers.append(nn.Linear(this_in, this_out))
            if not is_last:
                layers.append(nn.ReLU())
                if dropout_p is not None:
                    layers.append(nn.Dropout(dropout_p))
        self.layers = nn.Sequential(*layers)

    def forward(self, input):
        return self.layers(input)

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

        parser.add_argument('--selection_attention', action='store_true')
        parser.add_argument('--feed_context', action='store_true')
        parser.add_argument('--feed_context_attend', action='store_true')
        parser.add_argument('--feed_context_attend_separate', action='store_true')

        parser.add_argument('--no_word_attention',
                            action='store_true',
                            help="don't attend to the context in the word output layer")

        parser.add_argument('--untie_grus',
                            action='store_true',
                            help="don't use the same weights for the reader and writer")

        parser.add_argument('--word_attention_constrained',
                            action='store_true',
                            help="don't allow attention to dots that aren't mentioned in the utterance")

        parser.add_argument('--feed_attention_constrained',
                            action='store_true',
                            help="don't allow attention to dots that aren't mentioned in the utterance")

        parser.add_argument('--mark_dots_mentioned',
                            action='store_true',
                            help='give an indicator feature for whether the given dot should be mentioned')

        parser.add_argument('--attention_type', choices=['softmax', 'sigmoid'], default='softmax')

        parser.add_argument('--selection_beliefs', choices=['none', 'selected', 'partners'],
                            default='none', help='selected: indicator on what you chose. partners: indicator on what the other person has')

        parser.add_argument('--generation_beliefs', choices=['none', 'selected', 'partners'],
                            default='none', help='selected: indicator on what you chose. partners: indicator on what the other person has')

    def __init__(self, word_dict, args):
        super(RnnReferenceModel, self).__init__()

        domain = get_domain(args.domain)

        self.word_dict = word_dict
        self.args = args
        self.num_ent = domain.num_ent()

        # define modules:
        self.word_embed = nn.Embedding(len(self.word_dict), args.nembed_word)

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

        self.reader = nn.GRU(
            input_size=gru_input_size,
            hidden_size=args.nhid_lang,
            bias=True)

        self.writer = nn.GRUCell(
            input_size=gru_input_size,
            hidden_size=args.nhid_lang,
            bias=True)

        # TODO: refactor to add n_layers dimension
        # nhid_lang
        self.reader_init_h = torch.nn.Parameter(torch.zeros(args.nhid_lang), requires_grad=True)

        # tie the weights between reader and writer?
        if not args.untie_grus:
            self.writer.weight_ih = self.reader.weight_ih_l0
            self.writer.weight_hh = self.reader.weight_hh_l0
            self.writer.bias_ih = self.reader.bias_ih_l0
            self.writer.bias_hh = self.reader.bias_hh_l0
            self.writer_init_h = self.reader_init_h
        else:
            # n_layers * n_directions x nhid_lang
            self.writer_init_h = torch.nn.Parameter(torch.zeros(args.nhid_lang), requires_grad=True)

        # if args.attentive_selection_encoder:
        #     self.selection_attention = nn.Sequential(
        #
        #     )

        if args.no_word_attention:
            h2o_input_dim = args.nhid_lang
        else:
            h2o_input_dim = args.nhid_lang + args.nembed_ctx

        self.hid2output = nn.Sequential(
            nn.Linear(h2o_input_dim, args.nembed_word),
            nn.ReLU(),
            nn.Dropout(args.dropout),
            )

        if args.share_attn:
            assert not args.separate_attn
            # TODO: get rid of layers here
            assert not args.mark_dots_mentioned
            assert args.selection_beliefs == 'none'
            assert args.generation_beliefs == 'none'
            self.attn = FeedForward(2, args.nhid_lang + args.nembed_ctx, args.nhid_attn, 1, dropout_p = args.dropout)
            if self.args.feed_context_attend_separate:
                self.feed_attn = FeedForward(2, args.nhid_lang + args.nembed_ctx, args.nhid_attn, 1, dropout_p = args.dropout)
            else:
                self.feed_attn = None
        elif args.separate_attn:
            for attn_name in ['lang', 'sel', 'ref', 'feed']:
                input_dim = args.nhid_lang + args.nembed_ctx
                if args.mark_dots_mentioned and attn_name in ['lang', 'feed']:
                    # TODO: consider tiling the indicator feature across more dimensions
                    input_dim += 1
                if args.selection_beliefs != 'none' and attn_name == 'sel':
                    input_dim += 1
                if args.generation_beliefs != 'none' and attn_name in ['lang', 'feed']:
                    input_dim += 1
                setattr(
                    self,
                    self._attention_name(attn_name),
                    FeedForward(2, input_dim, args.nhid_attn, 1, dropout_p=args.dropout)
                )
        else:
            assert not args.mark_dots_mentioned
            assert args.selection_beliefs == 'none'
            assert args.generation_beliefs == 'none'

            self.attn = nn.Sequential(
                nn.Linear(args.nhid_lang + args.nembed_ctx, args.nhid_sel),
                nn.ReLU(),
                nn.Dropout(args.dropout))
            for attn_name in ['lang', 'sel', 'ref']:
                setattr(
                    self,
                    self._attention_name(attn_name),
                    FeedForward(1, args.nhid_sel, args.nhid_sel, 1, dropout_p=args.dropout)
                )
            if self.args.feed_context_attend_separate:
                # TODO: why separate hidden dim for this?
                self.feed_attn = FeedForward(1, args.nhid_sel, args.nhid_attn, 1, dropout_p=args.dropout)
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

    def _attention_name(self, name):
        return '{}_attn'.format(name)

    def _apply_attention(self, name, input):
        if self.args.share_attn:
            logit = self.attn(input)
        elif self.args.separate_attn:
            attn_module = getattr(self, self._attention_name(name))
            logit = attn_module(input)
        else:
            attn_module = getattr(self, self._attention_name(name))
            logit = attn_module(self.attn(input))
        return logit.squeeze(-1)

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

    def reference_resolution(self, ctx_h, outs_emb, ref_inpt):
        if ref_inpt is None:
            return None

        bsz = ctx_h.size(0)
        num_dots = ctx_h.size(1)

        # reshape
        ref_inpt = torch.transpose(ref_inpt, 0, 2).contiguous().view(-1, bsz)
        ref_inpt = ref_inpt.view(-1, bsz).unsqueeze(2)
        ref_inpt = ref_inpt.expand(-1, -1, outs_emb.size(2))

        # gather indices
        ref_inpt = torch.gather(outs_emb, 0, ref_inpt)
        # reshape
        ref_inpt = ref_inpt.view(3, -1, ref_inpt.size(1), ref_inpt.size(2))

        # this mean pools embeddings for the referent's start and end, as well as the end of the sentence it occurs in
        # take mean
        ref_inpt = torch.mean(ref_inpt, 0)

        ref_inpt = ref_inpt.unsqueeze(2).expand(-1, -1, num_dots, -1)
        ctx_h = ctx_h.unsqueeze(0).expand(ref_inpt.size(0), ref_inpt.size(1), ref_inpt.size(2), ctx_h.size(-1))

        ref_logit = self._apply_attention('ref', torch.cat([ref_inpt, ctx_h], 3))
        return ref_logit

    def selection(self, ctx_h, outs_emb, sel_idx, beliefs=None):
        # outs_emb: length x batch_size x dim

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
        sel_inpt = sel_inpt.expand(ctx_h.size(0), ctx_h.size(1), ctx_h.size(2))
        to_cat = [sel_inpt, ctx_h]
        if beliefs is not None:
            to_cat.append(beliefs)
        sel_logit = self._apply_attention('sel', torch.cat(to_cat, 2))
        return sel_logit

    def _language_conditioned_dot_attention(self, ctx_h, lang_hs, use_feed_attn, dots_mentioned, generation_beliefs):
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

        constrain_attention = False
        if use_feed_attn and vars(self.args).get('feed_attention_constrained', False):
            constrain_attention = True
        if (not use_feed_attn) and vars(self.args).get('word_attention_constrained', False):
            constrain_attention = True

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
        attn_logit = self._apply_attention(
            'feed' if use_feed_attn else 'lang',
            torch.cat([lang_h_expand, ctx_h_expand], 3)
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

    def _forward(self, ctx_h, inpt, ref_inpt, sel_idx, lens=None, lang_h=None, compute_sel_out=False, pack=False,
                 dots_mentioned=None, selection_beliefs=None, generation_beliefs=None):
        # ctx_h: bsz x num_dots x nembed_ctx
        # lang_h: num_layers*num_directions x bsz x nhid_lang
        bsz = ctx_h.size(0)
        seq_len = inpt.size(0)

        # print('inpt size: {}'.format(inpt.size()))

        lang_hs, last_h, feed_ctx_attn_prob = self._read(
            ctx_h, inpt, lang_h=lang_h, lens=lens, pack=pack, dots_mentioned=dots_mentioned,
            generation_beliefs=generation_beliefs,
        )

        # compute language output
        if vars(self.args).get('no_word_attention', False):
            outs = self.hid2output(lang_hs)
            ctx_attn_prob = None
        else:
            ctx_attn_prob = self._language_conditioned_dot_attention(
                ctx_h, lang_hs, use_feed_attn=False, dots_mentioned=dots_mentioned, generation_beliefs=generation_beliefs
            )
            ctx_h_expand = ctx_h.unsqueeze(0).expand(seq_len, -1, -1, -1)

            ctx_h_lang = torch.einsum("tbnd,tbn->tbd", (ctx_h_expand,ctx_attn_prob))
            # ctx_h_lang = torch.sum(torch.mul(ctx_h_expand, lang_prob.unsqueeze(-1)), 2)

            outs = self.hid2output(torch.cat([lang_hs, ctx_h_lang], 2))
        outs = F.linear(outs, self.word_embed.weight)
        outs = outs.view(-1, outs.size(2))

        # compute referents output
        # print('ref_inpt.size(): {}'.format(ref_inpt.size()))
        # print('outs_emb.size(): {}'.format(outs_emb.size()))
        ref_out = self.reference_resolution(ctx_h, lang_hs, ref_inpt)

        if compute_sel_out:
            # compute selection
            # print('sel_idx size: {}'.format(sel_idx.size()))
            sel_out = self.selection(ctx_h, lang_hs, sel_idx, beliefs=selection_beliefs)
        else:
            sel_out = None

        return outs, ref_out, sel_out, last_h, ctx_attn_prob, feed_ctx_attn_prob

    def forward(self, ctx, inpt, ref_inpt, sel_idx, lens, dots_mentioned, selection_beliefs, generation_beliefs):
        if selection_beliefs is not None:
            raise NotImplementedError("selection_belief for non-hierarchical model")
        ctx_h = self.ctx_encoder(ctx.transpose(0,1))
        outs, ref_out, sel_out, last_h, ctx_attn_prob, feed_ctx_attn_prob = self._forward(
            ctx_h, inpt, ref_inpt, sel_idx, lens=lens, lang_h=None, compute_sel_out=True, pack=False,
            dots_mentioned=dots_mentioned, selection_beliefs=selection_beliefs, generation_beliefs=generation_beliefs
        )
        return outs, ref_out, sel_out, ctx_attn_prob, feed_ctx_attn_prob

    def _context_for_feeding(self, ctx_h, lang_hs, dots_mentioned, generation_beliefs):
        if self.args.feed_context_attend:
            # bsz x num_dots
            feed_ctx_attn_prob = self._language_conditioned_dot_attention(
                ctx_h, lang_hs, use_feed_attn=True, dots_mentioned=dots_mentioned, generation_beliefs=generation_beliefs
            ).squeeze(0).squeeze(-1)
            # bsz x num_dots x nembed_ctx
            ctx_emb = self.feed_ctx_layer(ctx_h)
            ctx_emb = torch.einsum("bn,bnd->bd", (feed_ctx_attn_prob, ctx_emb))
        else:
            # bsz x nembed_ctx
            ctx_emb = self.feed_ctx_layer(ctx_h).mean(1)
            feed_ctx_attn_prob = None
        return ctx_emb, feed_ctx_attn_prob

    def _read(self, ctx_h, inpt, lang_h, lens=None, pack=False, dots_mentioned=None, generation_beliefs=None):
        # lang_h: num_layers * num_directions x batch x nhid_lang
        bsz = ctx_h.size(0)
        seq_len = inpt.size(0)
        num_dots = ctx_h.size(1)

        if lang_h is None:
            # lang_h = self._zero(1, bsz, self.args.nhid_lang)
            if hasattr(self, 'reader_init_h'):
                reader_lang_h = self.reader_init_h.unsqueeze(0).unsqueeze(1).expand(1, bsz, -1).contiguous()
                writer_lang_h = self.writer_init_h.unsqueeze(0).unsqueeze(1).expand(1, bsz, -1).contiguous()
            else:
                reader_lang_h = self.init_h.unsqueeze(0).unsqueeze(1).expand(1, bsz, -1).contiguous()
                writer_lang_h = reader_lang_h

        else:
            reader_lang_h, writer_lang_h = lang_h, lang_h

        # seq_len x batch_size x nembed_word
        dialog_emb = self.embed_dialogue(inpt)
        if self.args.feed_context:
            # seq_len x bsz x (nembed_word+nembed_ctx)
            ctx_emb, feed_ctx_attn_prob = self._context_for_feeding(
                ctx_h, writer_lang_h, dots_mentioned=dots_mentioned, generation_beliefs=generation_beliefs
            )
            dialog_emb = torch.cat((dialog_emb, ctx_emb.expand(seq_len, -1, -1)), dim=-1)
        else:
            feed_ctx_attn_prob = None
        if pack:
            dialog_emb_before_pack = dialog_emb
            assert lens is not None
            with set_temporary_default_tensor_type(torch.FloatTensor):
                dialog_emb = pack_padded_sequence(dialog_emb, lens.cpu(), enforce_sorted=False)
        # print('lang_h size: {}'.format(lang_h.size()))
        reader_lang_hs, reader_last_h = self.reader(dialog_emb, reader_lang_h)
        if vars(self.args).get('untie_grus', False):
            # todo: implement this?
            raise NotImplementedError("untie_grus")
            assert False
            # writer_lang_hs, writer_last_h = self.writer(dialog_emb)

        lang_hs, last_h = reader_lang_hs, reader_last_h

        if pack:
            lang_hs, _ = pad_packed_sequence(lang_hs, total_length=seq_len)
        return lang_hs, last_h, feed_ctx_attn_prob

    def read(self, ctx_h, inpt, lang_h, prefix_token='THEM:', dots_mentioned=None):
        # Add a 'THEM:' token to the start of the message
        prefix = self.word2var(prefix_token).unsqueeze(0)
        inpt = torch.cat([prefix, inpt])
        lang_hs, lang_h, feed_ctx_attn_prob = self._read(ctx_h, inpt, lang_h, lens=None, pack=False, dots_mentioned=dots_mentioned)
        return lang_hs, lang_h

    def word2var(self, word):
        x = torch.Tensor(1).fill_(self.word_dict.get_idx(word)).long()
        return Variable(x)

    def write(self, ctx_h, lang_h, max_words, temperature,
              start_token='YOU:', stop_tokens=data.STOP_TOKENS, force_words=None, dots_mentioned=None,
              generation_beliefs=None):
        # ctx_h: batch x num_dots x nembed_ctx
        # lang_h: batch x hidden
        bsz, _ = lang_h.size()
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

        outs = [inpt.unsqueeze(0)]
        logprobs = []
        lang_hs = []

        if self.args.feed_context:
            # add a time dimension
            ctx_emb, feed_ctx_attn_prob = self._context_for_feeding(ctx_h, lang_h.unsqueeze(1), dots_mentioned)
        else:
            ctx_emb, feed_ctx_attn_prob = None, None
        ctx_attn_probs = []
        top_words = []
        for word_ix in range(max_words):
            # embed
            inpt_emb = self.embed_dialogue(inpt)
            if ctx_emb is not None:
                # bsz x (nembed_word+nembed_ctx)
                inpt_emb = torch.cat((inpt_emb, ctx_emb), dim=-1)
            lang_h = self.writer(inpt_emb, lang_h)
            lang_hs.append(lang_h)

            if self.word_dict.get_word(inpt.data[0]) in stop_tokens:
                break

            # compute language output
            if vars(self.args).get('no_word_attention', False):
                out_emb = self.hid2output(lang_h)
            else:
                # compute attention for language output
                # bsz x num_dots x hidden
                # lang_h_expand = lang_h.unsqueeze(1).expand(-1, num_dots, -1)
                # lang_logit = self._apply_attention('lang', torch.cat([lang_h_expand, ctx_h], -1))
                # dot_attention2 = F.softmax(lang_logit, dim=1)

                # add a time dimension to lang_h
                ctx_attn_prob = self._language_conditioned_dot_attention(
                    ctx_h, lang_h.unsqueeze(0), use_feed_attn=False, dots_mentioned=dots_mentioned,
                    generation_beliefs=generation_beliefs
                )
                # remove the time dimension
                ctx_attn_prob = ctx_attn_prob.squeeze(0)
                ctx_attn_probs.append(ctx_attn_prob)
                # lang_prob = dot_attention2.expand_as(ctx_h)
                # ctx_h_lang = torch.sum(torch.mul(ctx_h, dot_attention2.expand_as(ctx_h)), 1)
                ctx_h_lang = torch.einsum("bnd,bn->bd", (ctx_h,ctx_attn_prob))
                out_emb = self.hid2output(torch.cat([lang_h, ctx_h_lang], 1))
            out = F.linear(out_emb, self.word_embed.weight)

            scores = out.div(temperature)
            scores = scores.sub(scores.max().item())

            mask = Variable(self.special_token_mask.to(scores.device))
            scores = scores.add(mask.unsqueeze(0))

            prob = F.softmax(scores, dim=-1)
            logprob = F.log_softmax(scores, dim=-1)

            top_words.append(logprob.topk(5, dim=-1))

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
        _, lang_h = self.read(ctx_h, outs, lang_h.unsqueeze(0), dots_mentioned=dots_mentioned)
        lang_h = lang_h.squeeze(0)

        extra = {
            'feed_ctx_attn_prob': feed_ctx_attn_prob,
            'word_ctx_attn_probs': ctx_attn_probs,
            'top_words': top_words,
        }
        if ctx_attn_probs:
            extra['word_ctx_attn_prob_mean'] = torch.mean(torch.stack(ctx_attn_probs, 0), 0)

        return outs, logprobs, lang_h, torch.cat(lang_hs, 0), extra


class HierarchicalRnnReferenceModel(RnnReferenceModel):
    corpus_ty = corpora.reference_sentence.ReferenceSentenceCorpus
    engine_ty = HierarchicalRnnReferenceEngine

    @classmethod
    def add_args(cls, parser):
        # args from RnnReferenceModel will be added separately
        pass

    def forward(self, ctx, inpts, ref_inpts, sel_idx, lens, dots_mentioned, selection_beliefs, generation_beliefs):
        # inpts is a list, one item per sentence
        # ref_inpts also a list, one per sentence
        # sel_idx is index into the last sentence in the dialogue

        ctx_h = self.ctx_encoder(ctx.transpose(0,1))
        lang_h = None

        all_outs = []
        all_ref_outs = []

        sel_out = None

        all_ctx_attn_prob = []
        all_feed_ctx_attn_prob = []

        if generation_beliefs is not None:
            assert len(generation_beliefs) == len(inpts)

        assert len(inpts) == len(ref_inpts)
        for i in range(len(inpts)):
            inpt = inpts[i]
            ref_inpt = ref_inpts[i]
            is_last = i == len(inpts) - 1
            this_lens = lens[i]
            outs, ref_out, sel_out, lang_h, ctx_attn_prob, feed_ctx_attn_prob = self._forward(
                ctx_h, inpt, ref_inpt, sel_idx, lens=this_lens, lang_h=lang_h, compute_sel_out=is_last, pack=True,
                dots_mentioned=dots_mentioned[i], selection_beliefs=selection_beliefs if is_last else None,
                generation_beliefs=generation_beliefs[i] if generation_beliefs is not None else None
            )
            all_outs.append(outs)
            all_ref_outs.append(ref_out)
            all_ctx_attn_prob.append(ctx_attn_prob)
            all_feed_ctx_attn_prob.append(feed_ctx_attn_prob)

        assert sel_out is not None

        return all_outs, all_ref_outs, sel_out, all_ctx_attn_prob, all_feed_ctx_attn_prob
