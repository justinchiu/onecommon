cp ../src/logs/gpt-only-5005-4/chat_state.db gpt.db
cp ../src/logs/baseline-only-5006-2/chat_state.db baseline.db
cp ../src/logs/human-only-5007-2/chat_state.db human.db

python dump_databases.py gpt.db baseline.db human.db --output_name blah.json

