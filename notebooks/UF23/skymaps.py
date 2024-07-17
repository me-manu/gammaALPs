import sys

sys.path.append('../notebooks')
sys.path.append('../gammaALPs')

from gammaALPs.core import Source, ALP, ModuleList
import numpy as np
import matplotlib.pyplot as plt

import healpy as hp
import time


models = ['base', 'expX', 'spur', 'neCL', 'twistX', 'nebCor', 'cre10', 'synCG', 'jansson12']

try:
    model = sys.argv[1]
except:
    model = 'base'

try:
    NSIDE = int(sys.argv[2])
except:
    NSIDE = 30

pix = np.arange(hp.nside2npix(NSIDE))  # get the pixels
ll, bb = hp.pixelfunc.pix2ang(NSIDE, pix, lonlat=True)  #  get the galactic coordinates for each pixel
# model = 'jansson12'

print(model)
print(NSIDE)
filename = f'notebooks/UF23/data/pggnew/new_pgg_{model}_{NSIDE}.npy'
# with open(filename, 'w') as file:
#     pass


pa_in = np.diag([0., 0., 1.])
EGeV = np.array([1000.])  # energy
pgg = np.zeros((pix.shape[0],EGeV.shape[0]))  # array to store the results
src = Source(z=0.1, ra=0., dec=0.)  # some random source for initialization

# coupling and mass at which we want to calculate the conversion probability:
g = 0.5
m = 10.

t1 = time.time()
for i, l in enumerate(ll):
    src.l = l
    src.b = bb[i]

    ml = ModuleList(ALP(m=m, g=g), src, pin=pa_in, EGeV=EGeV, log_level='warning')

    ml.add_propagation("GMF", 0, model="UF23", UF23_model=model)
    px, py, pa = ml.run()  # run the code

    pgg[i] = px + py  # save the result
    
    if i < ll.size - 1:
        del ml

np.save(filename, pgg)

t2 = time.time()
print ("It took", t2-t1, "seconds")

