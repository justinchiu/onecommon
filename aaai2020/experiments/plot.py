import argparse
import matplotlib.pyplot as plt
import pandas

STATS = ['train_ppl', 'train_select_acc', 'train_ref_acc', 'valid_ppl', 'valid_select_acc', 'valid_ref_acc']

#DEFAULT_STATS = ['valid_ppl']
DEFAULT_STATS = ['valid_ppl', 'valid_select_acc', 'valid_ref_acc']

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
                    key, value = item.split()
                    if key == 'epoch':
                        epoch = int(value)
                        if epoch not in stats_by_epoch:
                            stats_by_epoch[epoch] = {}
                    else:
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
        collected_df.plot(title=stat)
        plt.show()
