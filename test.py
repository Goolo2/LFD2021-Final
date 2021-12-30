import numpy as np
import random

for i in range(10):
    print(i)
    if random.randint(0, 5000) % 50000 == 0:
        print('yes')
        
print(0%50000)