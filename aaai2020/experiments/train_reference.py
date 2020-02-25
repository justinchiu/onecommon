import argparse
import pprint
import sys

import models
import utils
from domain import get_domain

import engines

def add_loss_args(parser):
    pass
    group = parser.add_argument_group('loss')
    group.add_argument('--lang_weight', type=float, default=1.0,
                        help='language loss weight')
    group.add_argument('--ref_weight', type=float, default=1.0,
                        help='reference loss weight')
    group.add_argument('--sel_weight', type=float, default=1.0,
                        help='selection loss weight')
    group.add_argument('--lang_only_self', action='store_true')

    # these args only make sense if --lang_only_self is True
    group.add_argument('--word_attention_supervised', action='store_true')
    group.add_argument('--feed_attention_supervised', action='store_true')

    group.add_argument('--attention_supervision_method', choices=['kl', 'penalize_unmentioned'], default='kl')

def main():
    parser = argparse.ArgumentParser(description='training script for reference resolution')
    parser.add_argument('--data', type=str, default='data/onecommon',
        help='location of the data corpus')
    parser.add_argument('--model_type', type=str, default='rnn_reference_model',
        help='type of model to use', choices=models.get_model_names())
    parser.add_argument('--ctx_encoder_type', type=str, default='mlp_encoder',
        help='type of context encoder to use', choices=models.get_ctx_encoder_names())

    parser.add_argument('--unk_threshold', type=int, default=20,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--temperature', type=float, default=0.1,
        help='temperature')
    parser.add_argument('--seed', type=int, default=1,
        help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--model_file', type=str,  default='tmp.th',
        help='path to save the final model')
    parser.add_argument('--domain', type=str, default='one_common',
        help='domain for the dialogue')
    parser.add_argument('--tensorboard_log', action='store_true', default=False,
        help='log training with tensorboard')
    parser.add_argument('--repeat_train', action='store_true', default=False,
        help='repeat training n times')
    parser.add_argument('--corpus_type', choices=['full', 'uncorrelated', 'success_only'], default='full',
        help='type of training corpus to use')


    engines.add_training_args(parser)
    add_loss_args(parser)
    models.add_model_args(parser)
    engines.add_engine_args(parser)

    utils.dump_git_status(sys.stdout)

    print(' '.join(sys.argv))
    args = parser.parse_args()
    pprint.pprint(vars(args))

    if args.repeat_train:
        seeds = list(range(10))
    else:
        seeds = [1]

    for seed in seeds:

        def model_filename_fn(name, extension):
            return '{}_{}_{}.{}'.format(args.model_file, seed, name, extension)

        utils.use_cuda(args.cuda)
        utils.set_seed(args.seed)

        domain = get_domain(args.domain)
        model_ty = models.get_model_type(args.model_type)

        corpus = model_ty.corpus_ty(domain, args.data,
                                    train='train_reference_{}.txt'.format(seed),
                                    valid='valid_reference_{}.txt'.format(seed),
                                    test='test_reference_{}.txt'.format(seed),
            freq_cutoff=args.unk_threshold, verbose=True)

        model = model_ty(corpus.word_dict, args)
        if args.cuda:
            model.cuda()

        engine = model_ty.engine_ty(model, args, verbose=True)
        if args.optimizer == 'adam':
            best_valid_loss, best_model = engine.train(corpus, model_filename_fn)
        elif args.optimizer == 'rmsprop':
            best_valid_loss, best_model = engine.train_scheduled(corpus, model_filename_fn)

        utils.save_model(best_model.cpu(), model_filename_fn('best', 'th'))
        utils.save_model(best_model.cpu().state_dict(), model_filename_fn('best', 'stdict'))


if __name__ == '__main__':
    main()
