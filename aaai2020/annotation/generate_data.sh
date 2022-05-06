#!/bin/bash

# python reference_annotation.py --output_brat_format --success_only --correct_misspellings --replace_strings
# python reference_annotation.py --output_brat_format --success_only --correct_misspellings --replace_strings --start_batch_index 30
# python reference_annotation.py --output_markable_annotation
# python reference_annotation.py --referent_aggregation

for seed in `seq 0 9`; do
  python transform_referents_to_txt.py --normalize --seed $seed
  python transform_train_to_context_txt.py --normalize --seed $seed
done
python transform_scenarios_to_txt.py --normalize --input_file ../experiments/data/onecommon/shared_4.json --output_file ../experiments/data/onecommon/shared_4.txt
python transform_scenarios_to_txt.py --normalize --input_file ../experiments/data/onecommon/shared_5.json --output_file ../experiments/data/onecommon/shared_5.txt
python transform_scenarios_to_txt.py --normalize --input_file ../experiments/data/onecommon/shared_6.json --output_file ../experiments/data/onecommon/shared_6.txt
