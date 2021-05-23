import numpy as np

def hash2intarray(h):
    b = []
    for i in range(len(h)//4):
        b.append(int(h[4*i:4*(i+1)], 16))
    return np.asarray(b, dtype=np.int64)

def intarray2hash(ia):
    c = list(map(lambda x: "{0:#0{1}x}".format(x,6).replace("0x",""), ia))
    return "".join(c)
