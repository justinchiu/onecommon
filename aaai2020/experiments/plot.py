import argparse
import matplotlib.pyplot as plt
import pandas

STATS = ['valid_ppl', 'valid_select_acc', 'valid_ref_acc']

DEFAULT_STATS = ['valid_ppl']

def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("log_files", nargs='+')
    parser.add_argument("--stats", choices=STATS, nargs='+', default=DEFAULT_STATS)
    return parser

def parse(log_file):
    stats_by_epoch = {}
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
    return stats_by_epoch

if __name__ == "__main__":
    args = make_parser().parse_args()
    dfs_by_name = {}
    for log_file in args.log_files:
        stats_by_epoch = parse(log_file)
        dfs_by_name[log_file] = pandas.DataFrame(stats_by_epoch).transpose()
    for stat in args.stats:
        collected_df = pandas.DataFrame({
            name: df[stat] for name, df in dfs_by_name.items()
        })
        collected_df.plot(title=stat)
        plt.show()
