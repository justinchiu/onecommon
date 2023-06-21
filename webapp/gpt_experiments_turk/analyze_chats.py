import numpy as np
import pandas
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import matplotlib.ticker as mtick
import nltk
#to_print = list(reversed(['uabaseline_same_opt', 'pragmatic_confidence', 'human']))
to_print = list(reversed(['gpt', 'pragmatic_confidence', 'human']))
# to_print = list(reversed(['uabaseline_same_opt', 'human']))
# to_print = list(reversed(['human']))

def analyze(all_chats, verbose=True, min_successful_games=None, min_completed_games=None, print_order=None, human_human_only=False):
    agent_type_counts = Counter()
    successful_agent_type_counts = Counter()
    completed_agent_type_counts = Counter()
    success = 0
    completed = 0
    if min_successful_games is not None or min_completed_games is not None:
        if min_successful_games is None:
            min_successful_games = 0
        if min_completed_games is None:
            min_completed_games = 0
        _worker_completed_counts, _worker_success_counts, _ = analyze(all_chats, verbose=False)
        _worker_hh_completed_counts, _worker_hh_success_counts, _ = analyze(all_chats, verbose=False, human_human_only=True)
        worker_filter = set([
            worker for worker in set(_worker_completed_counts.keys()) | set(_worker_success_counts.keys())
            if (_worker_completed_counts[worker] >= min_completed_games) and (_worker_success_counts[worker] >= min_successful_games)
        ])
    else:
        _worker_completed_counts = _worker_success_counts = None
        _worker_hh_completed_counts = _worker_hh_success_counts = None
        worker_filter = None

    worker_completed_counts = Counter()
    worker_success_counts = Counter()

    chat_stats = []

    for chat in all_chats:
        if worker_filter is not None:
            if chat['opponent_type'] == 'human' and len(chat['workers']) != 2:
                continue
            if chat['opponent_type'] != 'human' and len(chat['workers']) != 1:
                continue
            if all(worker not in worker_filter for worker in chat['workers']):
                continue
        if human_human_only and chat['opponent_type'] != 'human':
            continue
        chat = chat.copy()
        chat['num_turns'] = len(chat['dialogue'])
        chat['total_words'] = sum(len(d) for d in human_dialogues(chat))
        if chat['num_turns'] > 0:
            chat['words_per_turn'] = float(chat['total_words']) / chat['num_turns']
        opponent_type = chat['opponent_type']
        agent_type_counts[opponent_type] += 1
        if chat['outcome'] and chat['outcome'].get('reward', None) == 1:
            success += 1
            successful_agent_type_counts[chat['opponent_type']] += 1
            for worker in chat['workers']:
                worker_success_counts[worker] += 1
            chat['success'] = 1
        else:
            chat['success'] = 0
        if _worker_completed_counts is not None:
            wsc = np.array([_worker_success_counts[worker] for worker in chat['workers']])
            wcc = np.array([_worker_completed_counts[worker] for worker in chat['workers']])
            success_rates = wsc / wcc
            min_success_rate = np.min(success_rates)

            chat['worker_success_counts'] = wsc
            chat['worker_completed_counts'] = wcc
            chat['min_success_rate'] = min_success_rate

            whhsc = np.array([_worker_hh_success_counts[worker] for worker in chat['workers']])
            whhcc = np.array([_worker_hh_completed_counts[worker] for worker in chat['workers']])
            hh_success_rates = whhsc / whhcc
            min_hh_success_rate = np.min(hh_success_rates)
            chat['worker_human_human_success_counts'] = whhsc
            chat['worker_human_human_completed_counts'] = whhcc
            chat['min_human_human_success_rate'] = min_hh_success_rate
        if chat['num_players_selected'] == 2:
            if verbose:
                if len(chat['workers']) == 0:
                    print("0 workers found but both players selected for chat_id {}".format(chat['chat_id']))
                if chat['opponent_type'] == 'human' and len(chat['workers']) != 2:
                    print("{} workers found in a human game where both players selected for chat_id {}".format(
                        len(chat['workers']),
                        chat['chat_id'])
                    )
            for worker in chat['workers']:
                worker_completed_counts[worker] += 1
            completed += 1
            completed_agent_type_counts[chat['opponent_type']] += 1
            chat_stats.append(chat)

    def print_counts(counter):
        if print_order:
            print(','.join(print_order))
            print(','.join(str(counter[key]) for key in print_order))
        else:
            print(counter)
    if verbose:
        print("{} chats found".format(len(all_chats)))
        print_counts(agent_type_counts)
        print()
        print("{} completed".format(completed))
        print_counts(completed_agent_type_counts)
        print()
        print("{} successful".format(success))
        print_counts(successful_agent_type_counts)
        print()
        win_rates = {k: successful_agent_type_counts[k] / completed_agent_type_counts[k]
                     for k in set(completed_agent_type_counts.keys()) | set(successful_agent_type_counts.keys())}
        print("win rates")
        print_counts(win_rates)
        print()
    return worker_completed_counts, worker_success_counts, chat_stats

def tokenize_and_filter(utterance):
    punct = set(".,\"'!-")
    toks = nltk.word_tokenize(utterance)
    return [tok for tok in toks if tok not in punct]

def human_dialogues(chat):
    dialogue = chat['dialogue']
    human_agent_numbers = set([int(k) for k, v in chat['agent_types'].items()
                           if v == 'human'])
    return [tokenize_and_filter(d) for ix, d in dialogue if ix in human_agent_numbers]

if __name__ == "__main__":
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("chat_json_file")
    parser.add_argument("--min_successful_games", type=int)
    parser.add_argument("--min_completed_games", type=int)
    parser.add_argument("--min_success_type", choices=['all_games', 'human_human_games'])
    args = parser.parse_args()
    with open(args.chat_json_file, 'r') as f:
        chats = json.load(f)
    _, _, chat_stats = analyze(
        chats, min_successful_games=args.min_successful_games, min_completed_games=args.min_completed_games,
        print_order=to_print
        #print_order=['human']
        #print_order=None,
    )

    # name_lookups = {
    #     'human': 'Human',
    #     'uabaseline_same_opt': 'U&A\'20',
    #     'pragmatic_confidence': 'Full+Prag',
    # }

    name_lookups = {
        'human': 'Human',
        'pragmatic_confidence': 'Fried 2021',
        'gpt': 'Ours',
    }
    diffs = [('pragmatic_confidence', 'gpt')]

    chat_stats_df = pandas.DataFrame(chat_stats)
    x_vals = []
    means_by_key = {k: [] for k in to_print}
    std_errs_by_key = {k: [] for k in to_print}
    num_turns_by_key = {k: [] for k in to_print}
    num_turns_std_dev_by_key = {k: [] for k in to_print}
    num_words_per_turn_by_key = {k: [] for k in to_print}
    num_words_per_turn_std_dev_by_key = {k: [] for k in to_print}
    success_key, success_name = {
        'all_games': ('min_success_rate', 'Overall'),
        'human_human_games': ('min_human_human_success_rate', 'Human-Human')
    }[args.min_success_type]
    for min_success_rate in np.linspace(0, 1.0, 41):
        filtered = chat_stats_df[chat_stats_df[success_key] >= min_success_rate]
        success_counts = filtered.groupby('opponent_type')['success'].sum()
        success_rates = filtered.groupby('opponent_type')['success'].mean()
        num_turns = filtered.groupby('opponent_type')['num_turns'].mean()
        num_turns_std_dev = filtered.groupby('opponent_type')['num_turns'].std()
        counts = filtered.groupby('opponent_type')['success'].count()
        num_words_per_turn = filtered.groupby('opponent_type')['words_per_turn'].mean()
        num_words_per_turn_std_dev = filtered.groupby('opponent_type')['words_per_turn'].std()
        # https://stats.stackexchange.com/questions/29641/standard-error-for-the-mean-of-a-sample-of-binomial-random-variables
        standard_error_mean = np.sqrt(success_rates * (1 - success_rates) / counts)
        total_count = 0
        for k in to_print:
            total_count += counts[k]
            means_by_key[k].append(success_rates[k])
            std_errs_by_key[k].append(standard_error_mean[k])
            num_turns_by_key[k].append(num_turns[k])
            num_turns_std_dev_by_key[k].append(num_turns_std_dev[k])
            num_words_per_turn_by_key[k].append(num_words_per_turn[k])
            num_words_per_turn_std_dev_by_key[k].append(num_words_per_turn_std_dev[k])
        #print("{:.2f}: {}".format(min_success_rate, total_count))
        for diff in diffs:
            a, b = diff
            n = counts[a] + counts[b]
            p = (success_counts[a] + success_counts[b]) / (n)
            z = (success_rates[a] - success_rates[b]) / np.sqrt(p * (1 - p) * (1.0 / counts[a] + 1.0 / counts[b]))
            p_value = stats.t.sf(z, n-1)
            if p_value < 0.05:
                sig_str = "**"
            elif p_value < 0.1:
                sig_str = "*"
            else:
                sig_str = ""
            if len(diffs) > 1:
                print("{}{:.2f}\t{}\t{}\t{:.4f}\t{:.4f}".format(sig_str, min_success_rate, total_count, diff, z, p_value))
            else:
                print("{}{:.2f}\t{}\t{:.4f}\t{:.4f}".format(sig_str, min_success_rate, total_count, z, p_value))
        if len(diffs) > 1:
            print()
        x_vals.append(min_success_rate)
    x_vals = np.array(x_vals)
    colors = {
        'human': 'gold',
        'gpt': 'grey',
        'pragmatic_confidence': 'mediumblue',
    }
    print(num_turns_by_key)
    def plot(stats_by_key, std_errs_by_key=None, y_percentage=True):
        fig, ax = plt.subplots()
        for k, y_mean in stats_by_key.items():
            y_mean = np.array(y_mean)
            #plt.plot(x_vals, y_mean, label=name_lookups[k])
            #plt.errorbar(x_vals, y_mean, yerr=std_errs_by_key[k], label=name_lookups[k])
            p = ax.plot(x_vals, y_mean, label=name_lookups[k], color=colors[k])
            if std_errs_by_key is not None:
                sem = np.array(std_errs_by_key[k])
                lower=y_mean-sem
                upper=y_mean+sem
                ax.plot(x_vals, lower, color=colors[k], alpha=0.1)
                ax.plot(x_vals, upper, color=colors[k], alpha=0.1)
                ax.fill_between(x_vals, lower, upper, alpha=0.1, color=colors[k])
        ax.xaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        if y_percentage:
            ax.yaxis.set_major_formatter(mtick.PercentFormatter(1.0, decimals=0))
        if std_errs_by_key is not None:
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
    plot(means_by_key, std_errs_by_key)
    plt.ylabel("Per-Condition Success")
    plt.xlabel("Minimum {} Worker Success".format(success_name))
    plt.legend(loc='lower right')
    #plt.vlines(0.287, ymin=0.2, ymax=1)
    plt.show()

    #plot(num_turns_by_key, num_turns_std_dev_by_key, y_percentage=False)
    plot(num_turns_by_key, y_percentage=False)
    plt.ylabel("Number of Turns")
    plt.xlabel("Minimum Worker Success")
    plt.legend(loc='lower right')
    #plt.vlines(0.287, ymin=0.2, ymax=1)
    plt.show()

    plot(num_words_per_turn_by_key, num_words_per_turn_std_dev_by_key, y_percentage=False)
    plt.ylabel("Number of Words per Turn")
    plt.xlabel("Minimum Worker Success")
    plt.legend(loc='lower right')
    #plt.vlines(0.287, ymin=0.2, ymax=1)
    plt.show()
