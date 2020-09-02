import argparse
import matplotlib.pyplot as plt
import pandas

STATS = [
    '{}_{}'.format(split, stat)
    for split in ['train', 'valid']
    for stat in ['correct_ppl', 
                 'select_acc', 
                 'ref_acc', 'ref_f1', 'ref_exact_match', 
                 'partner_ref_acc', 'partner_ref_f1', 'partner_ref_exact_match',
                 'next_mention_acc', 'next_mention_f1', 'next_mention_exact_match',
                 'l1_loss',
                ]
]


#DEFAULT_STATS = ['valid_ppl']
DEFAULT_STATS = ['valid_correct_ppl', 'valid_select_acc', 'valid_ref_acc', 'valid_ref_f1', 'valid_ref_exact_match', 'valid_next_mention_f1', 'valid_next_mention_exact_match']

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_files", nargs='+')
    parser.add_argument("--stats", choices=STATS, nargs='+', default=DEFAULT_STATS)
    return parser

def parse(log_file):
    stats_by_epoch = {}
    try:
        with open(log_file, 'r') as f:
            for line in f:
                if not line.startswith('epoch '):
                    continue
                epoch = None
                for item in line.split('\t'):
                    try:
                        key, value = item.split()
                        if key == 'epoch':
                            epoch = int(value)
                            if epoch not in stats_by_epoch:
                                stats_by_epoch[epoch] = {}
                        else:
                            assert epoch is not None
                            stats_by_epoch[epoch][key] = float(value)
                    except:
                        tokens = item.split()
                        if tokens[0] == 'epoch':
                            epoch = int(tokens[1])
                            if epoch not in stats_by_epoch:
                                stats_by_epoch[epoch] = {}
                            tokens = tokens[2:]
                            if not tokens:
                                continue
                        if tokens[0] in ['train_ppl(avg', 'valid_ppl(avg']:
                            key = ' '.join(tokens[:3])
                            value = tokens[-1]
                        else:
                            key, value = tokens
                    key = key.replace("accuracy", "acc")
                    assert epoch is not None
                    stats_by_epoch[epoch][key] = float(value)
    except UnicodeDecodeError as e:
        print(e)
        return None
    return stats_by_epoch

if __name__ == "__main__":
    args = make_parser().parse_args()
    dfs_by_name = {}
    for log_file in args.log_files:
        stats_by_epoch = parse(log_file)
        if stats_by_epoch is None:
            print("error parsing {}".format(log_file))
        dfs_by_name[log_file] = pandas.DataFrame(stats_by_epoch).transpose()
    for stat in args.stats:
        data = {}
        for name, df in dfs_by_name.items():
            if stat in df.columns:
                data[name] = df[stat]
            else:
                print('df {} does not have stat {}'.format(name, stat))
        collected_df = pandas.DataFrame(data)
        if collected_df.empty:
            print("stat {} is empty".format(stat))
        else:
            collected_df.plot(title=stat)
        plt.show()
