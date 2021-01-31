import numpy as np
import pandas
from collections import Counter, defaultdict
import matplotlib.pyplot as plt
to_print = ['uabaseline_same_opt', 'pragmatic_confidence', 'human']

def analyze(all_chats, verbose=True, min_successful_games=None, min_completed_games=None, print_order=None):
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
        worker_filter = set([
            worker for worker in set(_worker_completed_counts.keys()) | set(_worker_success_counts.keys())
            if (_worker_completed_counts[worker] >= min_completed_games) and (_worker_success_counts[worker] >= min_successful_games)
        ])
    else:
        _worker_completed_counts = _worker_success_counts = None
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
        chat = chat.copy()
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

if __name__ == "__main__":
    import json
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("chat_json_file")
    parser.add_argument("--min_successful_games", type=int)
    parser.add_argument("--min_completed_games", type=int)
    args = parser.parse_args()
    with open(args.chat_json_file, 'r') as f:
        chats = json.load(f)
    _, _, chat_stats = analyze(
        chats, min_successful_games=args.min_successful_games, min_completed_games=args.min_completed_games,
        print_order=to_print
        #print_order=['human']
        #print_order=None,
    )

    name_lookups = {
        'human': 'Human',
        'uabaseline_same_opt': 'U&A\'20',
        'pragmatic_confidence': 'Full+Prag',
    }

    chat_stats_df = pandas.DataFrame(chat_stats)
    x_vals = []
    by_key = {k: [] for k in to_print}
    for min_success_rate in np.linspace(0, 1.0, 21):
        filtered = chat_stats_df[chat_stats_df['min_success_rate'] >= min_success_rate]
        success_rates = filtered.groupby('opponent_type')['success'].mean()
        counts = filtered.groupby('opponent_type')['success'].count()
        total_count = 0
        for k in to_print:
            total_count += counts[k]
            by_key[k].append(success_rates[k])
        print("{:.2f}: {}".format(min_success_rate, total_count))
        x_vals.append(min_success_rate)
    for k, y_vals in by_key.items():
        plt.plot(x_vals, y_vals, label=name_lookups[k])
    plt.ylabel("Per-Condition Success Rate")
    plt.xlabel("Minimum Worker Overall Success Rate")
    plt.legend()
    plt.vlines(0.287, ymin=0.2, ymax=1)
    plt.show()
