#!/bin/bash

export PYTHONPATH="../../aaai2020:../../aaai2020/experiments:../../aaai2020/experiments/models:..:$PYTHONPATH"

scenarios="shared_4"

python web/chat_app.py --port 3000 \
    --schema-path data/schema.json \
    --config web/app_params.json \
    --scenarios-path ../../aaai2020/experiments/data/onecommon/${scenarios}.json \
    --output web/debug \
    $@

