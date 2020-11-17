#!/bin/bash
export PYTHONPATH="../../aaai2020:../../aaai2020/experiments:../../aaai2020/experiments/models:..:$PYTHONPATH"

python web/chat_app.py --port 5000 \
    --schema-path data/schema.json \
    --config web/app_params.json --scenarios-path data/aaai_train_scenarios_2.json \
    --output web/sample
