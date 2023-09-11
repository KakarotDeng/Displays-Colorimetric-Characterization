import numpy
import random

with open('3.txt', 'w') as f:
    for i in range(150):
        a = random.uniform(0, 122.27)
        b = random.uniform(0, 122.47)
        c = random.uniform(0, 121.61)
        a1 = format(a, '.2f')
        b1 = format(b, '.2f')
        c1 = format(c, '.2f')
        f.write(a1)
        f.write('\t')
        f.write(b1)
        f.write('\t')
        f.write(c1)
        f.write('\n')






