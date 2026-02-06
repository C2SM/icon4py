import json
import numpy
import csv
import sys

if len(sys.argv) < 2:
    print("Usage: python read_gt4py_timers.py <input_file> [--csv]")
    sys.exit(1)

input_file = sys.argv[1]
data = json.load(open(input_file))

if len(sys.argv) > 2 and sys.argv[2] == '--csv':
    with open('output.csv', 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Function', 'Mean', 'Std'])
        for k, v in data.items():
            if v.get('metrics').get('compute'):
                arr = numpy.array(v.get('metrics').get('compute')[1:])
                writer.writerow([k.split('<')[0], arr.mean(), arr.std()])
else:
    for k, v in data.items():
        if v.get('metrics').get('compute'):
            arr = numpy.array(v.get('metrics').get('compute')[1:])
            print(f"{k.split('<')[0]}: Mean = {arr.mean()}, Std = {arr.std()}")
