# %matplotlib inline 

# import pydicom library

import pydicom

# import matplotlib and numpy

import matplotlib.pyplot as plt 
import matplotlib.image as mpimage
import numpy as np 

# import operating system and glob libraries

import os, glob

# import some useful date functions

from datetime import datetime

ds1 = pydicom.read_file("test.dcm")

level = ds1.WindowCenter
window = ds1.WindowWidth
# ...or set window/level manually to values you want
vmin = level - window/2
vmax = level + window/2
plt.imshow(ds1.pixel_array, cmap='gray', vmin=vmin, vmax=vmax)
plt.show()
