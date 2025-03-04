import argparse
from collections import defaultdict
import json
import sqlite3
from datetime import datetime
import os
import shutil
import warnings
import atexit
from gevent.pywsgi import WSGIServer
# ablate this
# ablate apache
import sys

from cocoa.core.scenario_db import add_scenario_arguments, ScenarioDB
from cocoa.core.schema import Schema
from cocoa.core.util import read_json
from cocoa.systems.human_system import HumanSystem
from cocoa.web.main.logger import WebLogger

from core.scenario import Scenario
from systems import get_system
from systems.rnn_system import CUDA
from main.db_reader import DatabaseReader
from main.backend import DatabaseManager

from flask import g
from web.main.backend import Backend
from flask import Flask, current_app
from flask_socketio import SocketIO
socketio = SocketIO()

import pdb

# TODO: verify
from utils import use_cuda
if CUDA:
    use_cuda(True)

DB_FILE_NAME = 'chat_state.db'
LOG_FILE_NAME = 'log.out'
ERROR_LOG_FILE_NAME = 'error_log.out'
TRANSCRIPTS_DIR = 'transcripts'

def close_connection(exception):
    backend = getattr(g, '_backend', None)
    if backend is not None:
        backend.close()

def create_app(debug=False, templates_dir='templates'):
    """Create an application."""

    app = Flask(__name__, template_folder=os.path.abspath(templates_dir), static_url_path='/dialogue/static')
    app.debug = debug
    app.config['SECRET_KEY'] = 'gjr39dkjn344_!67#'
    app.config['PROPAGATE_EXCEPTIONS'] = True

    from web.views.action import action
    from web.views.admin import admin
    from web.views.annotation import annotation
    from web.views.coreference import coreference
    from web.views.selfplay import selfplay
    from cocoa.web.views.chat import chat
    app.register_blueprint(chat, url_prefix='/dialogue')
    app.register_blueprint(action, url_prefix='/dialogue')
    app.register_blueprint(admin, url_prefix='/dialogue')
    app.register_blueprint(annotation, url_prefix='/dialogue')
    app.register_blueprint(coreference, url_prefix='/dialogue')
    app.register_blueprint(selfplay, url_prefix='/dialogue')

    app.teardown_appcontext_funcs = [close_connection]

    socketio.init_app(app)
    return app
######################

def add_website_arguments(parser):
    parser.add_argument('--port', type=int, default=5000,
                        help='Port to start server on')
    parser.add_argument('--host', type=str, default='127.0.0.1',
                        help='Host IP address to run app on. Defaults to localhost.')
    parser.add_argument('--config', type=str, default='app_params.json',
                        help='Path to JSON file containing configurations for website')
    parser.add_argument('--output', type=str,
                        default="web/output/{}".format(datetime.now().strftime("%Y-%m-%d")),
                        help='Name of directory for storing website output (debug and error logs, chats, '
                             'and database). Defaults to a web_output/current_date, with the current date formatted as '
                             '%%Y-%%m-%%d". '
                             'If the provided directory exists, all data in it is overwritten unless the '
                             '--reuse parameter is provided.')
    parser.add_argument('--reuse', action='store_true', help='If provided, reuses the existing database file in the '
                                                             'output directory.')


def add_systems(args, config_dict, schema):
    """
    Params:
    config_dict: A dictionary that maps the bot name to a dictionary containing configs for the bot. The
        dictionary should contain the bot type (key 'type') and. for bots that use an underlying model for generation,
        the path to the directory containing the parameters, vocab, etc. for the model.
    Returns:
    agents: A dict mapping from the bot name to the System object for that bot.
    pairing_probabilities: A dict mapping from the bot name to the probability that a user is paired with that
        bot. Also includes the pairing probability for humans (backend.Partner.Human)
    """

    total_probs = 0.0
    systems = {HumanSystem.name(): HumanSystem()}
    pairing_probabilities = {}
    timed = False if params['debug'] else True
    for (sys_name, info) in config_dict.items():
        if "active" not in info.keys():
            warnings.warn("active status not specified for bot %s - assuming that bot is inactive." % sys_name)
        if info["active"]:
            name = info["type"]
            try:
                model_path = info["model_path"]
                markable_detector_path = info["markable_detector_path"]
                model = get_system(
                    name, args, schema=schema, timed=timed,
                    model_path=model_path, markable_detector_path=markable_detector_path,
                    inference_args=info["inference_args"],
                )
            except ValueError:
                warnings.warn(
                    'Unrecognized model type in {} for configuration '
                    '{}. Ignoring configuration.'.format(info, sys_name))
                continue
            systems[sys_name] = model
            if 'prob' in info.keys():
                prob = float(info['prob'])
                pairing_probabilities[sys_name] = prob
                total_probs += prob

    if total_probs > 1.0:
        raise ValueError("Probabilities for active bots can't exceed 1.0.")
    if len(pairing_probabilities.keys()) != 0 and len(pairing_probabilities.keys()) != len(systems.keys()):
        remaining_prob = (1.0-total_probs)/(len(systems.keys()) - len(pairing_probabilities.keys()))
    else:
        remaining_prob = 1.0 / len(systems.keys())
    inactive_bots = set()
    for system_name in systems.keys():
        if system_name not in pairing_probabilities.keys():
            if remaining_prob == 0.0:
                inactive_bots.add(system_name)
            else:
                pairing_probabilities[system_name] = remaining_prob

    for sys_name in inactive_bots:
        systems.pop(sys_name, None)

    if args.no_human_partner:
        pairing_probabilities[HumanSystem.name()] = 0
        # renormalize
        Z = sum(pairing_probabilities.values())
        pairing_probabilities = {k: v / Z for k,v in pairing_probabilities.items()}

    return systems, pairing_probabilities

def cleanup(flask_app):
    db_path = flask_app.config['user_params']['db']['location']
    transcript_path = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'transcripts.json')
    accepted_transcripts = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'accepted-transcripts.json')
    review_path = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'reviews.json')
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    DatabaseReader.dump_chats(cursor, flask_app.config['scenario_db'], transcript_path, accepted_only=False, include_turk=True)
    DatabaseReader.dump_chats(cursor, flask_app.config['scenario_db'], accepted_transcripts, accepted_only=True, include_turk=True)
    if flask_app.config['user_params']['end_survey'] == 1:
        surveys_path = os.path.join(flask_app.config['user_params']['logging']['chat_dir'], 'surveys.json')
        DatabaseReader.dump_surveys(cursor, surveys_path)
    DatabaseReader.dump_reviewed_chat(cursor, review_path)
    conn.close()

def init(output_dir, reuse=False, rewrite=False):
    db_file = os.path.join(output_dir, DB_FILE_NAME)
    log_file = os.path.join(output_dir, LOG_FILE_NAME + datetime.now().strftime("-%Y-%m-%d-%H-%M-%S"))
    error_log_file = os.path.join(output_dir, ERROR_LOG_FILE_NAME + datetime.now().strftime("-%Y-%m-%d-%H-%M-%S"))
    transcripts_dir = os.path.join(output_dir, TRANSCRIPTS_DIR)
    # TODO: don't remove everything
    if not reuse:
        if False:
            print("removing everything not implemented!")
            sys.exit(1)
        else:
            if os.path.exists(output_dir):
                shutil.rmtree(output_dir)
            os.makedirs(output_dir)

            db = DatabaseManager.init_database(db_file)

            if os.path.exists(transcripts_dir):
                shutil.rmtree(transcripts_dir)
            os.makedirs(transcripts_dir)
    else:
        db = DatabaseManager(db_file)

    return db, log_file, error_log_file, transcripts_dir


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-scenarios', type=int)
    parser.add_argument('--starting-scenario', type=int)
    parser.add_argument('--visualize-transcripts', type=str, default='data/final_transcripts.json')
    parser.add_argument('--markable_annotation', type=str, default='data/markable_annotation.json')
    parser.add_argument('--batch_info', type=str, default='data/batch_info.json')
    parser.add_argument('--referent_annotation', type=str, default='data/referent_annotation.json')
    parser.add_argument('--rejected_referent_annotation', type=str, default='data/rejected_referent_annotation.json')
    parser.add_argument('--aggregated_referent_annotation', type=str, default='data/aggregated_referent_annotation.json')
    parser.add_argument('--model_referent_annotation', type=str, default='data/model_referent_annotation.json')
    parser.add_argument('--selfplay_scenarios', type=str, default='data/shared_5.json')
    parser.add_argument('--selfplay_markables', type=str, default='data/selfplay_markables.json')
    parser.add_argument('--selfplay_referents', type=str, default='data/selfplay_referents.json')
    parser.add_argument('--dump-only', action='store_true')
    parser.add_argument('--no_human_partner', action='store_true')
    add_website_arguments(parser)
    add_scenario_arguments(parser)
    print(' '.join(sys.argv))
    args = parser.parse_args()

    if not args.reuse and os.path.exists(args.output):
        overwrite = input("[warning] overwriting data: Continue? [Y]:")
        if not overwrite == "Y":
            sys.exit()
    elif args.reuse and not os.path.exists(args.output):
        raise ValueError("output directory does not exist (can not be reused)")

    params_file = args.config
    with open(params_file) as fin:
        params = json.load(fin)

    db, log_file, error_log_file, transcripts_dir = init(args.output, args.reuse)
    error_log_file = open(error_log_file, 'w')

    print(params)

    WebLogger.initialize(log_file)
    params['db'] = {}
    params['db']['location'] = db.db_file
    params['logging'] = {}
    params['logging']['app_log'] = log_file
    params['logging']['chat_dir'] = transcripts_dir

    if 'task_title' not in params.keys():
        raise ValueError("Title of task should be specified in config file with the key 'task_title'")

    instructions = None
    if 'instructions' in params.keys():
        instructions_file = open(params['instructions'], 'r')
        instructions = "".join(instructions_file.readlines())
        instructions_file.close()
    else:
        raise ValueError("Location of file containing instructions for task should be specified in config with the key "
                         "'instructions")

    coreference_instructions = None
    if 'coreference_instructions' in params.keys():
        coreference_instructions_file = open(params['coreference_instructions'], 'r')
        coreference_instructions = "".join(coreference_instructions_file.readlines())
        coreference_instructions_file.close()
    else:
        raise ValueError("Location of file containing coreference instructions for task should be specified in config with the key "
                         "'coreference_instructions")

    templates_dir = None
    if 'templates_dir' in params.keys():
        templates_dir = params['templates_dir']
    else:
        raise ValueError("Location of HTML templates should be specified in config with the key templates_dir")
    if not os.path.exists(templates_dir):
            raise ValueError("Specified HTML template location doesn't exist: %s" % templates_dir)

    app = create_app(debug=params['debug'], templates_dir=templates_dir)

    schema_path = args.schema_path

    if not os.path.exists(schema_path):
        raise ValueError("No schema file found at %s" % schema_path)

    schema = Schema(schema_path)
    scenarios = read_json(args.scenarios_path)
    if args.starting_scenario is not None:
        scenarios = scenarios[args.starting_scenario:]
    if args.num_scenarios is not None:
        scenarios = scenarios[:args.num_scenarios]
    print("num scenarios: {}".format(len(scenarios)))
    scenario_db = ScenarioDB.from_dict(schema, scenarios, Scenario)
    app.config['scenario_db'] = scenario_db

    if 'models' not in params.keys():
        params['models'] = {}

    if 'quit_after' not in params.keys():
        params['quit_after'] = params['status_params']['chat']['num_seconds'] + 500

    if 'skip_chat_enabled' not in params.keys():
        params['skip_chat_enabled'] = False

    if 'end_survey' not in params.keys() :
        params['end_survey'] = 0

    if 'debug' not in params:
        params['debug'] = False

    systems, pairing_probabilities = add_systems(args, params['models'], schema)

    db.add_scenarios(scenario_db, systems, update=args.reuse)
    #add_scenarios_to_db(db_file, scenario_db, systems, update=args.reuse)

    app.config['systems'] = systems
    app.config['sessions'] = defaultdict(None)
    app.config['pairing_probabilities'] = pairing_probabilities
    app.config['num_chats_per_scenario'] = params.get('num_chats_per_scenario', {k: 1 for k in systems})
    for k in systems:
        assert k in app.config['num_chats_per_scenario']
    app.config['schema'] = schema
    app.config['user_params'] = params
    app.config['controller_map'] = defaultdict(None)
    app.config['instructions'] = instructions
    app.config['coreference_instructions'] = coreference_instructions
    app.config['task_title'] = params['task_title']

    if not os.path.exists(args.visualize_transcripts):
        raise ValueError("Final transcripts not found")
    with open(args.visualize_transcripts, "r") as f:
        chat_data = json.load(f)
    app.config['chat_data'] = chat_data

    if not os.path.exists(args.markable_annotation):
        raise ValueError("Markable annotation not found")
    with open(args.markable_annotation, "r") as f:
        markable_annotation = json.load(f)
    app.config['markable_annotation'] = markable_annotation

    if not os.path.exists(args.batch_info):
        raise ValueError("Batch info not found")
    with open(args.batch_info, "r") as f:
        batch_info = json.load(f)
    app.config['batch_info'] = batch_info

    if not os.path.exists(args.referent_annotation):
        referent_annotation = {}
    else:
        with open(args.referent_annotation, "r") as f:
            referent_annotation = json.load(f)

    if not os.path.exists(args.aggregated_referent_annotation):
        aggregated_referent_annotation = {}
    else:
        with open(args.aggregated_referent_annotation, "r") as f:
            aggregated_referent_annotation = json.load(f)
    if not os.path.exists(args.model_referent_annotation):
        model_referent_annotation = {}
    else:
        with open(args.model_referent_annotation, "r") as f:
            model_referent_annotation = json.load(f)

    if not os.path.exists(args.selfplay_scenarios):
        raise ValueError("Selfplay scenarios not found")
    else:
        selfplay_scenarios = read_json(args.selfplay_scenarios)
    if not os.path.exists(args.selfplay_markables):
        selfplay_markables = {}
    else:
        with open(args.selfplay_markables, "r") as f:
            selfplay_markables = json.load(f)
    if not os.path.exists(args.selfplay_referents):
        selfplay_referents = {}
    else:
        with open(args.selfplay_referents, "r") as f:
            selfplay_referents = json.load(f)

    app.config['referent_annotation'] = referent_annotation
    app.config['referent_annotation_save_path'] = args.referent_annotation
    app.config['aggregated_referent_annotation'] = aggregated_referent_annotation
    app.config['model_referent_annotation'] = model_referent_annotation
    app.config['selfplay_scenarios'] = selfplay_scenarios
    app.config['selfplay_markables'] = selfplay_markables
    app.config['selfplay_referents'] = selfplay_referents
    
    if not os.path.exists(args.rejected_referent_annotation):
        rejected_referent_annotation = {}
    else:
        with open(args.rejected_referent_annotation, "r") as f:
            rejected_referent_annotation = json.load(f)
    app.config['rejected_referent_annotation'] = rejected_referent_annotation
    app.config['rejected_referent_annotation_save_path'] = args.rejected_referent_annotation

    if 'icon' not in params.keys():
        app.config['task_icon'] = 'handshake.jpg'
    else:
        app.config['task_icon'] = params['icon']


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

    print("App setup complete")

    if args.dump_only:
        cleanup(app)
    else:
        atexit.register(cleanup, flask_app=app)
        #app.run('0.0.0.0', debug=True, port=args.port, ssl_context=('/home/ubuntu/server.crt', '/home/ubuntu/server.key'))
        # server = WSGIServer(('', args.port), app, log=WebLogger.get_logger(), error_log=error_log_file)
        server = WSGIServer(
            ('', args.port), app, log=WebLogger.get_logger(), error_log=error_log_file,
                            #keyfile='/etc/letsencrypt/live/berkeleynlp.com/privkey.pem',
                            #certfile='/etc/letsencrypt/live/berkeleynlp.com/fullchain.pem'
                            # justin linux
                            #keyfile='/home/justinchiu/keys/key.pem',
                            #certfile='/home/justinchiu/keys/cert.pem',
                            # justin mac
                            #keyfile='/Users/justinchiu/keys/key.pem',
                            #certfile='/Users/justinchiu/keys/cert.pem',
                            ## aws
                            #keyfile='/home/ubuntu/keys/key.pem',
                            #certfile='/home/ubuntu/keys/cert.pem',
        )
        server.serve_forever()
