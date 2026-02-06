import json
import numpy
import csv
import sys

if len(sys.argv) < 2:
    print("Usage: python print_gt4py_timers.py <input_file> [--csv]")
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
                if len(arr) > 0:
                    mean = arr.mean()
                    if not numpy.isnan(mean):
                        writer.writerow([k.split('<')[0], mean, arr.std()])
else:
    for k, v in data.items():
        if v.get('metrics').get('compute'):
            arr = numpy.array(v.get('metrics').get('compute')[1:])
            if len(arr) > 0:
                mean = arr.mean()
                if not numpy.isnan(mean):
                    print(f"{k.split('<')[0]}: Mean = {mean}, Std = {arr.std()}")
