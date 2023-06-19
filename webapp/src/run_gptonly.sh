#!/bin/bash

port=5005
iter=5

# think aaai2020 is for annotation, maybe?
export PYTHONPATH="..:.:../../aaai2020:../../aaai2020/experiments:..:$PYTHONPATH"

scenarios="shared_4"
instance=gpt-only-${port}-${iter}

python web/chat_app.py --port ${port} \
    --schema-path data/schema.json \
    --config web/app_params.json.gptonly \
    --scenarios-path ../../aaai2020/experiments/data/onecommon/${scenarios}.json \
    --output logs/${instance} \
    --no_human_partner \
    $@
