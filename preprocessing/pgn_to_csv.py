with open('test3.csv', 'w') as t:
    with open('test2.pgn', 'r') as f:
        while True:
            line1 = f.readline()
            line2 = f.readline()
            if not line1:
                break

            line1 = line1.split(']')
            line2 = line2.strip('{ ').strip(' }\n')

            t.write(f'{line2},{line1[0][9:]}\n')
