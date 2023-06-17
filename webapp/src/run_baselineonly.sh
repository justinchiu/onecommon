#!/bin/bash

port=5006
iter=1

# think aaai2020 is for annotation, maybe?
export PYTHONPATH="..:.:../../aaai2020:../../aaai2020/experiments:..:$PYTHONPATH"

scenarios="shared_4"
#instance=turk3
#instance=lab3-oldgpt
instance=baseline-only-${port}-${iter}
#port=5007

python web/chat_app.py --port ${port} \
    --schema-path data/schema.json \
    --config web/app_params.json.baselineonly \
    --scenarios-path ../../aaai2020/experiments/data/onecommon/${scenarios}.json \
    --output logs/${instance} \
    --no_human_partner \
    $@
