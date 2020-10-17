import argparse
import pprint
import sys
import os

import models
import utils
from domain import get_domain

import engines
from engines.rnn_reference_engine import add_loss_args


def main():
    parser = argparse.ArgumentParser(description='training script for reference resolution')
    parser.add_argument('--data', type=str, default='data/onecommon',
        help='location of the data corpus')
    parser.add_argument('--max_instances_per_split', type=int)
    parser.add_argument('--max_mentions_per_utterance', type=int)
    parser.add_argument('--crosstalk_split', choices=[0, 1], default=None, type=int)

    parser.add_argument('--model_type', type=str, default='rnn_reference_model',
        help='type of model to use', choices=models.get_model_names())
    parser.add_argument('--ctx_encoder_type', type=str, default='mlp_encoder',
        help='type of context encoder to use', choices=models.get_ctx_encoder_names())

    parser.add_argument('--unk_threshold', type=int, default=20,
        help='minimum word frequency to be in dictionary')
    parser.add_argument('--temperature', type=float, default=0.1,
        help='temperature')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--cuda', action='store_true', default=False,
        help='use CUDA')
    parser.add_argument('--model_file', type=str,  default='default', help='name to use in model saving')
    parser.add_argument('--output_dir', type=str, default='expts')
    parser.add_argument('--domain', type=str, default='one_common',
        help='domain for the dialogue')
    parser.add_argument('--tensorboard_log', action='store_true', default=False,
        help='log training with tensorboard')
    parser.add_argument('--repeat_train', action='store_true', default=False,
        help='repeat training n times')
    parser.add_argument('--fold_nums', nargs='*', type=int)
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

    if args.fold_nums:
        fold_nums = args.fold_nums
    else:
        if args.repeat_train:
            fold_nums = list(range(10))
        else:
            fold_nums = [1]

    if args.output_dir:
        model_output_dir = os.path.join(args.output_dir, args.model_file)
    else:
        model_output_dir = args.model_file

    for fold_num in fold_nums:

        os.makedirs(model_output_dir, exist_ok=True)

        def model_filename_fn(name, extension):
            return os.path.join(
                model_output_dir,
                f'{fold_num}_{name}.{extension}',
            )

        utils.use_cuda(args.cuda)
        utils.set_seed(args.seed)

        domain = get_domain(args.domain)
        model_ty = models.get_model_type(args.model_type)

        corpus = model_ty.corpus_ty(
            domain, args.data,
            train='train_reference_{}.txt'.format(fold_num),
            valid='valid_reference_{}.txt'.format(fold_num),
            test='test_reference_{}.txt'.format(fold_num),
            freq_cutoff=args.unk_threshold, verbose=True,
            max_instances_per_split=args.max_instances_per_split,
            max_mentions_per_utterance=args.max_mentions_per_utterance,
            crosstalk_split=args.crosstalk_split,
        )

        model = model_ty(corpus.word_dict, args)
        if args.cuda:
            model.cuda()

        engine = model_ty.engine_ty(model, args, verbose=True)
        if args.optimizer == 'adam':
            best_valid_loss, best_model = engine.train(corpus, model_filename_fn)
        elif args.optimizer == 'rmsprop':
            best_valid_loss, best_model = engine.train_scheduled(corpus, model_filename_fn)

        utils.save_model(best_model.cpu(), model_filename_fn('best', 'th'), prefix_dir=None)
        utils.save_model(best_model.cpu().state_dict(), model_filename_fn('best', 'stdict'), prefix_dir=None)

        model.flatten_parameters()
        utils.save_model(model.cpu(), model_filename_fn(f'ep-{args.max_epoch}', 'th'), prefix_dir=None)
        utils.save_model(model.cpu().state_dict(), model_filename_fn(f'ep-{args.max_epoch}', 'stdict'), prefix_dir=None)


if __name__ == '__main__':
    main()
