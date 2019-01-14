import numpy as np
import time
def multi_list(a,b):
    c=[]
    for i in a:
        c.append(i*b[i])
    return c

def multi_array(a,b):
    c=a*b
    return c

a=range(1000)
b=range(1000)

time_start=time.time()
for i in range(10000):
    c=multi_list(a,b)
time_end=time.time()
print('multi list totally cost',time_end-time_start)

a=np.array(a,np.int16)
b=np.array(b,np.int16)
c=time_start=time.time()
for i in range(10000):
    multi_array(a,b)
time_end=time.time()
print('multi array totally cost',time_end-time_start)