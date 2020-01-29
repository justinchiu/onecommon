from models.ctx_encoder import MlpContextEncoder, AttentionContextEncoder, RelationalAttentionContextEncoder
from models.rnn_reference_model import RnnReferenceModel

MODELS = {
    'rnn_reference_model': RnnReferenceModel,
}

CTX_ENCODERS = {
    'mlp_encoder': MlpContextEncoder,
    'attn_encoder': AttentionContextEncoder,
    'rel_attn_encoder': RelationalAttentionContextEncoder,
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