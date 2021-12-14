import sys
if __name__ == "__main__":
    with open(sys.argv[1], 'r') as f:
        for line in f:
            if line.startswith('===') or line.startswith('dialog_len') or line.startswith('---'):
                continue
            if line.startswith('dialog '):
                line = '<h1> {} </h1>'.format(line.strip())
            if line.startswith('Agreement!') or line.startswith('Disagreement?!'):
                line = '<br>' + line.strip()
            if line.startswith('scenario_id'):
                line = line.strip() + '<br>'
            line = line.replace("<eos>", "<br>")
            line = line.replace("<selection>", "<br>")
            line = line.replace('width="430" height="430"', 'width="215" height="215"')
            print(line.strip())
