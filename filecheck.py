import numpy as np
import os
import glob

# Check that all of the files were made successfully.

froot  = "/Users/salvesen/outreach/asom/expansion/results/frames/expansion_"
files  = glob.glob(froot + "*")
Nfiles = len(files)
Nz     = np.ceil(np.log10(Nfiles)).astype(int)
kBmin  = 800.0
for i in np.arange(Nfiles):

    # Current filename
    file = froot + str(i).zfill(Nz) + ".png"

    # Does the file exist?
    if not os.path.isfile(file): print "/nThis file is missing: ", str(i).zfill(Nz)
    
    # Is the file too small?
    sizekB = os.stat(file).st_size / 1e3
    if (sizekB < kBmin): print "/nThis file is too small: ", str(i).zfill(Nz)

