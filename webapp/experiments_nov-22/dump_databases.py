import argparse
import sqlite3
import json

from cocoa.web.main.db_reader import DatabaseReader

from analyze_chats import analyze

def check_completed_info(cursor, chat_id):
    cursor.execute('SELECT * FROM event WHERE chat_id=? ORDER BY time ASC', (chat_id,))
    logged_events = cursor.fetchall()

    agent_select = {0: False, 1: False}

    for row in logged_events:
        agent, action, time, data = [row[k] for k in ('agent', 'action', 'time', 'data')]
        if action == 'select':
            agent_select[agent] = True
    return agent_select

def get_worker_ids(cursor, chat_id):
    cursor.execute('SELECT worker_id FROM mturk_task WHERE chat_id=?', (chat_id,))
    return [row[0] for row in cursor.fetchall()]

# def get_chat_outcome(cursor, chat_id):
#     cursor.execute('SELECT outcome FROM chat WHERE chat_id=?', (chat_id,))
#     outcome = cursor.fetchone()[0]
#     try:
#         outcome = json.loads(outcome)
#     except ValueError:
#         outcome = {'reward': -1}
#     return outcome

# def get_chat_agent_types(cursor, chat_id):
#     """Get types of the two agents in the chat specified by chat_id.

#     Returns:
#         {0: agent_name (str), 1: agent_name (str)}

#     """
#     try:
#         cursor.execute('SELECT agent_types FROM chat WHERE chat_id=?', (chat_id,))
#         agent_types = cursor.fetchone()[0]
#         agent_types = json.loads(agent_types)
#     except sqlite3.OperationalError:
#         agent_types = {0: 'human', 1: 'human'}
#     return agent_types


def survey_result(args, cursor, chat_id):
    cursor.execute('''SELECT * FROM survey where chat_id=?''', (chat_id, ))
    res = cursor.fetchone()
    if res:
        survey = True
        humanlike = res[3]
        comments = res[4]
        confidence = res[5]
        understand_you = res[6]
        understand_them = res[7]
    else:
        survey = False
        humanlike = None
        comments = None
        confidence = None
        understand_you = None
        understand_them = None
    return {
        'has_survey': survey,
        'humanlike': humanlike,
        'comments': comments,
        'confidence': confidence,
        'understand_you': understand_you,
        'understand_them': understand_them,
    }

def get_chats(args, db_name):
    records = []
    conn = sqlite3.connect(db_name)
    conn.row_factory = sqlite3.Row
    with conn:
        cursor = conn.cursor()
        cursor.execute('SELECT DISTINCT chat_id FROM event')
        ids = [x[0] for x in cursor.fetchall()]
        for chat_id in ids:
            outcome = DatabaseReader.get_chat_outcome(cursor, chat_id)
            agent_types = DatabaseReader.get_chat_agent_types(cursor, chat_id)
            this_opponent_agents = set(agent_types.values()) - {'human'}
            if not this_opponent_agents:
                opponent_agent = 'human'
            else:
                opponent_agent = next(iter(this_opponent_agents))
            # if outcome['reward'] == 1:
            #     num_success += 1
            #     success_by_opponent[opponent_agent] += 1
            # else:
            #     num_fail += 1
            #     fail_by_opponent[opponent_agent] += 1
            scenario_uuid = DatabaseReader.get_chat_scenario_id(cursor, chat_id)
            events = [e.to_dict() for e in DatabaseReader.get_chat_events(cursor, chat_id, include_meta=True)]

            num_selections = sum(check_completed_info(cursor, chat_id).values())

            chat = []
            select_id = {}
            for chat_event in events:
                if chat_event['action'] == 'message':
                    chat.append((chat_event['agent'], chat_event['data']))
            record = {
                'chat_id': chat_id,
                'outcome': outcome,
                'agent_types': agent_types,
                'opponent_type': opponent_agent,
                'dialogue': chat,
                'events': events,
                'scenario_id': scenario_uuid,
                'survey_result': survey_result(args, cursor, chat_id),
                'num_players_selected': num_selections,
                'workers': list(set(get_worker_ids(cursor, chat_id))),
            }
            records.append(record)
    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("db_names", nargs="+")
    parser.add_argument("--output_name")
    args = parser.parse_args()
    all_chats = []
    for db_name in args.db_names:
        all_chats += get_chats(args, db_name)
    if args.output_name:
        with open(args.output_name, 'w') as f:
            json.dump(all_chats, f, indent=4, sort_keys=True)
    analyze(all_chats)
