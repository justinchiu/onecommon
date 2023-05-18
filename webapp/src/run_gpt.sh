#!/bin/bash

# think aaai2020 is for annotation, maybe?
export PYTHONPATH="..:.:../../aaai2020:../../aaai2020/experiments:..:$PYTHONPATH"

scenarios="shared_4"
instance=0

python web/chat_app.py --port 5005 \
    --schema-path data/schema.json \
    --config web/app_params.json.gpt \
    --scenarios-path ../../aaai2020/experiments/data/onecommon/${scenarios}.json \
    --output logs/gpt-${instance} \
    $@
