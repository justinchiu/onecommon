from flask import Blueprint, jsonify, render_template, request, redirect, url_for, Markup, Response
from flask import current_app as app

from collections import defaultdict

from functools import wraps
import json
import sqlite3
import pdb

import time

from collections import defaultdict
import operator

from cocoa.web.views.utils import userid, format_message
from cocoa.web.main.utils import Status
from cocoa.core.event import Event

from main.db_reader import DatabaseReader

from web.main.backend import Backend
get_backend = Backend.get_backend

admin = Blueprint('admin', __name__)


@admin.route('/_accept/', methods=['GET'])
def accept():
    chat_id = request.args.get('chat_id')
    review_status = request.args.get('review_status')

    db_path = app.config['user_params']['db']['location']
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if int(review_status) >= 0:
        cursor.execute(
            ''' UPDATE review
            SET accept=1
            where chat_id=?''', (chat_id,))
    else:
        cursor.execute(
            '''INSERT INTO review VALUES (?,?, "")''', (chat_id, 1))
    conn.commit()
    return jsonify(success=True)

@admin.route('/_reject/', methods=['GET'])
def reject():
    chat_id = request.args.get('chat_id')
    review_status = request.args.get('review_status')
    reject_message = request.args.get('reject_message')

    db_path = app.config['user_params']['db']['location']
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    if int(review_status) >= 0:
        cursor.execute(
            ''' UPDATE review
            SET accept=0, message=?
            where chat_id=?''', (reject_message, chat_id,))
    else:
        cursor.execute(
            '''INSERT INTO review VALUES (?,?,?)''', (chat_id, 0, reject_message))
    conn.commit()
    return jsonify(success=True)

@admin.route('/_zero_active/', methods=['GET'])
def zero_active():
    try:
        db_path = app.config['user_params']['db']['location']
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('''UPDATE scenario SET active="[]"''')
        conn.commit()  
    except sqlite3.IntegrityError:
            print("WARNING: Rolled back transaction")
            return jsonify(success=False)
    return jsonify(success=True)

def _accept(cursor, chat_id):
    cursor.execute(
            '''INSERT INTO review VALUES (?,?, "")''', (chat_id, 1))
    return True

def _reject(cursor, chat_id, reject_message):
    cursor.execute(
            '''INSERT INTO review VALUES (?,?,?)''', (chat_id, 0, reject_message))
    return True


def check_auth(username, password):
    """This function is called to check if a username /
    password combination is valid.
    """
    return username == 'admin' and password == 'pragmatix'

def authenticate():
    """Sends a 401 response that enables basic auth"""
    return Response(
    'Could not verify your access level for that URL.\n'
    'You have to login with proper credentials', 401,
    {'WWW-Authenticate': 'Basic realm="Login Required"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

def clean_incomplete_chats(cursor):
    cursor.execute('SELECT DISTINCT chat_id FROM event')
    ids = [x[0] for x in cursor.fetchall()]
    for chat_id in ids:
        if not DatabaseReader.check_completed_info(cursor, chat_id):
            cursor.execute('DELETE FROM event where chat_id=?',(chat_id, ))

@admin.route('/admin')
@requires_auth
def visualize():
    backend = get_backend()
    db_path = app.config['user_params']['db']['location']
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # create table review if not exists
    if not cursor.execute('''SELECT name FROM sqlite_master WHERE type='table' AND name='review' ''').fetchone():
        cursor.execute(
            '''CREATE TABLE review (chat_id text, accept integer, message text)'''
        )
        conn.commit()
        print("created new table review")

    if not request.args.get('chat_id'):
        cursor.execute('SELECT DISTINCT chat_id FROM event')
        # cursor.execute('SELECT DISTINCT chat_id FROM chat')
        ids = [x[0] for x in cursor.fetchall()]
        cursor.execute('SELECT DISTINCT chat_id FROM review')
        reviewed_ids = [x[0] for x in cursor.fetchall()]
        print(request.args)
        print("num ids: {}".format(len(set(ids))))
        if 'incomplete' in request.args:
            # list only chat_ids which are incomplete and not reviewed yet
            ids = [chat_id for chat_id in ids if (chat_id not in reviewed_ids) and (not DatabaseReader.check_completed_info(cursor, chat_id))]
        elif 'completed' in request.args:
            # list all chat_ids which are completed
            ids = [chat_id for chat_id in ids if DatabaseReader.check_completed_info(cursor, chat_id)]
        else:
            # list only chat_ids which are completed but not reviewed yet
            ids = [chat_id for chat_id in ids if chat_id not in reviewed_ids and DatabaseReader.check_completed_info(cursor, chat_id)]
        outcomes = []
        dialogues = []
        scenario_ids = []
        opponent_agents = []
        num_success = 0
        num_fail = 0
        num_incomplete = 0

        success_by_opponent = defaultdict(int)
        fail_by_opponent = defaultdict(int)
        incomplete_by_opponent = defaultdict(int)

        for chat_id in ids:
            outcome = DatabaseReader.get_chat_outcome(cursor, chat_id)
            this_opponent_agents = set(DatabaseReader.get_chat_agent_types(cursor, chat_id).values()) - {'human'}
            if not this_opponent_agents:
                opponent_agent = 'human'
            else:
                opponent_agent = next(iter(this_opponent_agents))
            opponent_agents.append(opponent_agent)

            outcomes.append(outcome)
            if outcome['reward'] == 1:
                num_success += 1
                success_by_opponent[opponent_agent] += 1
            else:
                num_fail += 1
                fail_by_opponent[opponent_agent] += 1
            chat_info = DatabaseReader.get_chat_example(cursor, chat_id, app.config['scenario_db'], include_meta=True).to_dict()
            chat_text = ""
            select_id = {}
            for chat_event in chat_info['events']:
                if chat_event['action'] == 'message':
                    chat_text += "{}: {}\n".format(chat_event['agent'], chat_event['data'].encode('ascii', 'ignore'))
            dialogues.append(chat_text)
            scenario_ids.append(chat_info['scenario_uuid'])

        num_ids = len(ids)
        opponent_types = list(sorted(set(opponent_agents)))

        return render_template('chat_list.html',
                               num_chats = num_ids,
                               chat_ids = ids,
                               chat_outcomes = outcomes,
                               reviewed = ["" for i in range(num_ids)],
                               base_url = request.base_url + '?chat_id=',
                               num_success=num_success,
                               num_fail=num_fail,
                               num_incomplete=num_incomplete,
                               num_accept=0,
                               num_reject=0,
                               dialogues=dialogues,
                               opponent_agents=opponent_agents,
                               success_by_opponent=success_by_opponent,
                               fail_by_opponent=fail_by_opponent,
                               opponent_types=opponent_types,
                               scenario_ids=scenario_ids,
                              )
    else:
        chat_id = request.args.get('chat_id')
        chat_info = DatabaseReader.get_chat_example(cursor, chat_id, app.config['scenario_db'], include_meta=True).to_dict()
        chat_text = ""
        select_id = {}
        for chat_event in chat_info['events']:
            if chat_event['action'] == 'message':
                chat_text += "{}: {}\n".format(chat_event['agent'], chat_event['data'].encode('ascii', 'ignore'))
            elif chat_event['action'] == 'select':
                chat_text += "<{} selected {}>\n".format(chat_event['agent'], chat_event['data'])
                select_id[chat_event['agent']] = chat_event['data']

        cursor.execute('''SELECT accept, message FROM review where chat_id=?''', (chat_id, ))
        res = cursor.fetchone()
        if res:
            review_status = res[0]
            message = res[1]
        else:
            review_status = -1
            message = ""

        select = {}
        for agent in [0,1]:
            select[agent] = None
            if agent in select_id:
                for obj in chat_info['scenario']['kbs'][agent]:
                    if obj['id'] == select_id[agent]:
                        select[agent] = obj
                        break

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

        return render_template('visualize.html',
                                chat_id=chat_id,
                                chat_text=chat_text,
                                kb_0=chat_info['scenario']['kbs'][0],
                                kb_1=chat_info['scenario']['kbs'][1],
                                select_0=select[0],
                                select_1=select[1],
                                review_status=review_status,
                                message=message,
                                agent_0=chat_info['agents']['0'],
                                agent_1=chat_info['agents']['1'],
                                survey=survey,
                                humanlike=humanlike,
                                confidence=confidence,
                                comments=comments,
                               understand_you=understand_you,
                               understand_them=understand_them,
                                )

