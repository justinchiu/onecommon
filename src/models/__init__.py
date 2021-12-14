from models.ctx_encoder import MlpContextEncoder, AttentionContextEncoder, RelationalAttentionContextEncoder, RelationalAttentionContextEncoder2, RelationalAttentionContextEncoder3
from models.rnn_reference_model import RnnReferenceModel, HierarchicalRnnReferenceModel
# from models.ctx_encoder_ua import AttentionContextEncoder as AttentionContextEncoderUA
# from models.rnn_reference_model_ua import RnnReferenceModel as RnnReferenceModelUA

MODELS = {
    'rnn_reference_model': RnnReferenceModel,
    # 'rnn_reference_model_ua': RnnReferenceModelUA,
    'hierarchical_rnn_reference_model': HierarchicalRnnReferenceModel,
}

CTX_ENCODERS = {
    'mlp_encoder': MlpContextEncoder,
    'attn_encoder': AttentionContextEncoder,
    # 'attn_encoder_ua': AttentionContextEncoderUA,
    'rel_attn_encoder': RelationalAttentionContextEncoder,
    'rel_attn_encoder_2': RelationalAttentionContextEncoder2,
    'rel_attn_encoder_3': RelationalAttentionContextEncoder3,
}

def add_model_args(parser):
    group = parser.add_argument_group('model')
    for models in [MODELS, CTX_ENCODERS]:
        for model in models.values():
            model.add_args(group)

def get_model_names():
    return MODELS.keys()


def get_model_type(name):
    return MODELS[name]

def get_ctx_encoder_names():
    return CTX_ENCODERS.keys()

def get_ctx_encoder_type(name):
    return CTX_ENCODERS[name]
