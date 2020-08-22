import string

import pyro.ops
import torch
from torch import nn

BIG_NEG = -1e9


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


class AttentionLayer(nn.Module):
    @classmethod
    def add_args(cls, parser):
        pass

    def __init__(self, args, n_hidden_layers, input_dim, hidden_dim, dropout_p, language_dim):
        super().__init__()
        self.args = args
        self.feedforward = FeedForward(n_hidden_layers, input_dim, hidden_dim, output_dim=1, dropout_p=dropout_p)

    def forward(self, lang_input, input, ctx_differences, num_markables):
        # takes ctx_differences and num_markables as an argument for compatibility with StructuredAttentionLayer
        return self.feedforward(input).squeeze(-1), None, None


class StructuredAttentionLayer(nn.Module):
    @classmethod
    def add_args(cls, parser):
        parser.add_argument('--structured_attention_hidden_dim', type=int, default=64)
        parser.add_argument('--structured_attention_dropout', type=float, default=0.2)
        parser.add_argument('--structured_attention_marginalize', dest='structured_attention_marginalize', action='store_true')
        parser.add_argument('--structured_attention_no_marginalize', dest='structured_attention_marginalize', action='store_false')
        parser.add_argument('--structured_attention_language_conditioned', action='store_true')
        parser.set_defaults(structured_attention_marginalize=True)

    def __init__(self, args, n_hidden_layers, input_dim, hidden_dim, dropout_p, language_dim):
        super().__init__()
        self.args = args
        self.feedforward = FeedForward(n_hidden_layers, input_dim, hidden_dim, output_dim=2, dropout_p=dropout_p)
        self.relation_dim = 4 + 1 # +1 for distance

        self.num_ent = 7

        input_dim = self.relation_dim
        if self.args.structured_attention_language_conditioned:
            input_dim += language_dim

        self.relation_encoder = FeedForward(
            n_hidden_layers=1, input_dim=input_dim, hidden_dim=args.structured_attention_hidden_dim,
            output_dim=3,
            dropout_p=args.structured_attention_dropout,
        )

        # self.contraction_string = self.build_contraction_string(self.num_ent)

    def build_contraction_string(self, num_ent):
        var_names = string.ascii_lowercase[:num_ent]
        batch_name = 'z'
        assert batch_name not in var_names

        unary_factor_names = []
        binary_factor_names = []

        for i in range(num_ent):
            unary_factor_names.append(batch_name + var_names[i])
            for j in range(num_ent):
                if i >= j:
                    continue
                binary_factor_names.append(batch_name + var_names[i] + var_names[j])

        marginals = ','.join('{}{}'.format(batch_name, v) for v in var_names)
        joint = '{}{}'.format(batch_name, ''.join(var_names))
        output_factor_names = '{},{}'.format(joint,marginals)

        return '{}->{}'.format(','.join(unary_factor_names+binary_factor_names), output_factor_names)

    def forward(self, lang_input, input, ctx_differences, num_markables, normalize_joint=True):
        # takes num_markables as an argument for compatibility with StructuredTemporalAttentionLayer
        # max instances per batch (aka N) x batch_size x num_dots x hidden_dim
        contraction_string = self.build_contraction_string(self.num_ent)

        if input.dim() == 3:
            input = input.unsqueeze(0)
            expanded = True
        else:
            assert input.dim() == 4
            expanded = False

        N = input.size(0)
        bsz = input.size(1)
        unary_potentials = self.feedforward(input)
        assert bsz == ctx_differences.size(0)

        # ctx_differences: batch_size x (7*6/2=21) x relation_dim
        # batch_size x 21 x 3
        if vars(self.args).get('structured_attention_language_conditioned', False):
            num_pairs = ctx_differences.size(1)
            lang_input_expand = lang_input.unsqueeze(2).repeat_interleave(num_pairs, dim=2)
            ctx_differences_expand = ctx_differences.unsqueeze(0).repeat_interleave(N, dim=0)
            binary_potentials = self.relation_encoder(
                torch.cat((lang_input_expand, ctx_differences_expand), -1)
            )
        else:
            binary_potentials = self.relation_encoder(ctx_differences).unsqueeze(0).repeat_interleave(N, dim=0)

        assert N == binary_potentials.size(0)
        assert bsz == binary_potentials.size(1)
        num_pairs = binary_potentials.size(2)
        assert binary_potentials.size(-1) == 3

        # flatten time and batch
        unary_potentials = unary_potentials.view(N*bsz, unary_potentials.size(2), unary_potentials.size(3))
        binary_potentials = binary_potentials.view(N*bsz, binary_potentials.size(2), binary_potentials.size(3))

        # get a symmetric edge potential matrix
        # [a, b, c] -> [[a, b], [b, c]]
        # bsz x num_pairs x 4
        binary_potentials = torch.einsum(
            "brx,yx->bry",
            binary_potentials,
            torch.FloatTensor([[1,0,0],[0,1,0],[0,1,0],[0,0,1]]).to(binary_potentials.device)
        )
        # flatten time and batch and reshape the last dimension (size 4)  to a 2x2
        binary_potentials = binary_potentials.view(N*bsz, num_pairs, 2, 2)

        # transpose the batch and dot dimension so that we can unpack along dots
        unary_factors = unary_potentials.transpose(0,1)
        # transpose the batch and dot-pair dimension so that we can unpack along dot-pairs
        binary_factors = binary_potentials.transpose(0,1)

        outputs = pyro.ops.contract.einsum(
            contraction_string,
            *unary_factors,
            *binary_factors,
            modulo_total=True,
            backend='pyro.ops.einsum.torch_log',
        )
        joint_logits = outputs[0]
        marginal_logits = outputs[1:]

        assert len(marginal_logits) == self.num_ent

        # bsz x num_ent x 2
        marginal_logits = torch.stack(marginal_logits, dim=1)
        log_marginals = marginal_logits.log_softmax(dim=-1)
        # go from log probs for positive and negative to log odds
        assert log_marginals.size(-1) == 2
        marginal_log_probs = log_marginals.select(dim=-1,index=1) - log_marginals.select(dim=-1,index=0)
        assert marginal_log_probs.size() == (bsz*N, self.num_ent)
        marginal_log_probs = marginal_log_probs.view(N, bsz, self.num_ent)


        joint_log_probs = joint_logits.reshape(N, bsz, -1)
        if normalize_joint:
            joint_log_probs = joint_log_probs.log_softmax(dim=-1)
        assert joint_log_probs.size() == (N, bsz, 2**self.num_ent)
        joint_log_probs = joint_log_probs.view((N, bsz) + (2,) * self.num_ent)

        if expanded:
            assert N == 1
            joint_log_probs = joint_log_probs.squeeze(0)
            marginal_log_probs = marginal_log_probs.squeeze(0)

        return marginal_log_probs, joint_log_probs, None


class StructuredTemporalAttentionLayer(StructuredAttentionLayer):
    def __init__(self, args, n_hidden_layers, input_dim, hidden_dim, dropout_p, language_dim):
        super().__init__(args, n_hidden_layers, input_dim, hidden_dim, dropout_p, language_dim)
        if args.structured_temporal_attention_transitions == 'dot_id':
            self.self_transition_params = nn.Parameter(torch.zeros(3))
            # TODO: consider just fixing this to zeros
            self.other_transition_params = nn.Parameter(torch.zeros(3))
        elif args.structured_temporal_attention_transitions == 'none':
            self.transition_params = nn.Parameter(torch.zeros(3), requires_grad=False)
        elif args.structured_temporal_attention_transitions == 'relational':
            self.self_transition_params = nn.Parameter(torch.zeros(3))

            input_dim = self.relation_dim
            if self.args.structured_attention_language_conditioned:
                input_dim += language_dim
            self.temporal_relation_encoder = FeedForward(
                n_hidden_layers=1, input_dim=input_dim, hidden_dim=args.structured_attention_hidden_dim,
                output_dim=3,
                dropout_p=args.structured_attention_dropout,
            )
        else:
            raise NotImplementedError(f"--structured_temporal_attention_transitions={args.structured_temporal_attention_transitions}")

    def build_temporal_contraction_string(self, num_ent):
        assert num_ent * 2 < len(string.ascii_lowercase)
        var_names_1 = string.ascii_lowercase[:num_ent]
        var_names_2 = string.ascii_lowercase[num_ent:2*num_ent]

        batch_name = 'z'
        assert batch_name not in var_names_1 + var_names_2

        binary_factor_names = []

        for a in var_names_1:
            for b in var_names_2:
                binary_factor_names.append(batch_name + a + b)

        output_factor_name = batch_name+var_names_1+var_names_2
        return '{}->{}'.format(','.join(binary_factor_names), output_factor_name)

    def forward(self, lang_input, input, ctx_differences, num_markables):
        from torch_struct import LinearChainNoScanCRF
        marginal_log_probs, joint_logits, _ = super().forward(lang_input, input, ctx_differences, num_markables, normalize_joint=False)
        N, bsz, *dot_dims = joint_logits.size()
        assert N == num_markables.max()
        assert num_markables.dim() == 1 and num_markables.size(0) == bsz

        joint_log_probs = joint_logits.view(N, bsz, -1).contiguous().log_softmax(-1).view_as(joint_logits)

        if N <= 1:
            # TODO: fix this hack so that we have a consistent return semantics
            return marginal_log_probs, joint_log_probs, None

        num_dots = len(dot_dims)
        exp_num_dots = 2**num_dots
        # batch x num_markables x exp_num_dots
        emission_potentials = joint_logits.view(N, bsz, exp_num_dots).transpose(0, 1)

        # N-1 x bsz x exp_num_dots x exp_num_dots
        transition_potentials = self.make_transitions(lang_input, bsz, num_dots, ctx_differences)
        # bsz x N-1 x exp_num_dots x exp_num_dots
        transition_potentials = transition_potentials.transpose(0,1)
        edge_potentials = self.make_potentials(transition_potentials, emission_potentials)
        dist = LinearChainNoScanCRF(edge_potentials, lengths=num_markables)
        return marginal_log_probs, joint_log_probs, dist

    def make_transitions(self, lang_input, batch_size, num_dots, ctx_differences):
        N = lang_input.size(0)
        assert batch_size == lang_input.size(1)
        # 1(N-1) x 1(batch size) x 1(num pairs) x 3
        if self.args.structured_temporal_attention_transitions == 'none':
            transition_params = self.transition_params.unsqueeze(0).unsqueeze(1).unsqueeze(2)
        elif self.args.structured_temporal_attention_transitions == 'dot_id':
            transition_params = [
                (self.self_transition_params if i == j else self.other_transition_params)
                for i in range(num_dots)
                for j in range(num_dots)
            ]
            transition_params = torch.stack(transition_params, 0).unsqueeze(0).unsqueeze(1)
        elif self.args.structured_temporal_attention_transitions == 'relational':
            transition_params = torch.zeros(N-1, batch_size, num_dots*num_dots, 3)
            if vars(self.args).get('structured_attention_language_conditioned', False):
                num_pairs = ctx_differences.size(1)
                input_diffs = lang_input[1:] - lang_input[:-1]
                input_diffs_expand = input_diffs.unsqueeze(2).repeat_interleave(num_pairs, dim=2)
                ctx_differences_expand = ctx_differences.unsqueeze(0).repeat_interleave(N-1, dim=0)
                rel_encoded = self.temporal_relation_encoder(torch.cat((input_diffs_expand, ctx_differences_expand), -1))
            else:
                rel_encoded = self.temporal_relation_encoder(ctx_differences).unsqueeze(0).repeat_interleave(N-1, dim=0)
            def get_index(i, j):
                return i * num_dots + j
            ix = 0
            for i in range(num_dots):
                transition_params[:,:,get_index(i,i)] = self.self_transition_params
                for j in range(i+1, num_dots):
                    transition_params[:,:,get_index(i,j)] = rel_encoded[:,:,ix]
                    transition_params[:,:,get_index(j,i)] = rel_encoded[:,:,ix]
                    ix += 1
            assert ix == rel_encoded.size(2)
        else:
            raise NotImplementedError(self.args.structured_temporal_attention_transitions)

        # get a symmetric edge potential matrix
        # [a, b, c] -> [[a, b], [b, c]]
        num_potentials = num_dots * num_dots

        # TODO: this is currently inefficient, duplicating along the batch dimension before the contractions.
        #  but will be necessary if we have the transitions depend on the dot properties
        # bsz x num_potentials x 2 x 2
        binary_potentials = torch.einsum(
            "nbrx,yx->nbry",
            transition_params,
            torch.FloatTensor([[1,0,0],[0,1,0],[0,1,0],[0,0,1]]).to(transition_params.device)
        ).view(transition_params.size(0), transition_params.size(1), transition_params.size(2), 2, 2).expand(N-1, batch_size, num_potentials, 2, 2).view((N-1)*batch_size, num_potentials, 2, 2)

        # transpose the batch and dot-pair dimension so that we can unpack along dot-pairs
        binary_factors = binary_potentials.transpose(0,1)

        contraction_string = self.build_temporal_contraction_string(self.num_ent)
        # bsz x 2 x 2 x ... [num_ent*2 2s]
        output_factor = pyro.ops.contract.einsum(
            contraction_string,
            *binary_factors,
            modulo_total=True,
            backend='pyro.ops.einsum.torch_log',
        )[0]
        transition_potentials = output_factor.contiguous().view(N-1, batch_size, 2**self.num_ent, 2**self.num_ent)
        return transition_potentials

    def make_potentials(self, transition_potentials, emission_potentials):
        # following https://github.com/harvardnlp/pytorch-struct/blob/b6816a4d436136c6711fe2617995b556d5d4d300/torch_struct/linearchain.py#L137
        # transition_mat: batch x N x (2**num_dots) x (2**num_dots)
        # emission_potentials: bsz x num_markables x 2 x 2 x ... [num_dots 2s]
        bsz, N, exp_num_dots = emission_potentials.size()
        assert transition_potentials.size() == (bsz, N-1, exp_num_dots, exp_num_dots)

        # batch x num_mentions x C(to) x C(from)
        edge_potentials = torch.zeros((bsz, N-1, exp_num_dots, exp_num_dots), device=emission_potentials.device)
        edge_potentials[:,:,:,:] += transition_potentials.view(bsz,N-1,exp_num_dots,exp_num_dots)
        edge_potentials[:,:,:,:] += emission_potentials.view(bsz, N, exp_num_dots, 1)[:, 1:]
        edge_potentials[:,0,:,:] += emission_potentials.view(bsz, N, 1, exp_num_dots)[:, 0]

        return edge_potentials
