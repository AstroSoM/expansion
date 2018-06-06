import numpy as np
import h5py
from risset import *
from sphere import *  # <-- Commented out because it takes awhile to load (need it for video frames)
from decimal import Decimal

# Hubble constant [km s^-1 Mpc^-1]
H0 = 70.49

#====================================================================================================
# GENERATE THE AUDIO WAV FILE

# "Tag" for the output files
#tag = "fone"
#tag = "fcomb"
#tag = "risset"
tag = "test"

# Instantiate the risset class
f0    = 16.0   # This is our starting frequency [-] @ t = 0
N5th  = 192    # Number of musical 5th intervals to ascend over the full frequency sweep
Nrep  = 2#550    # Number of times to repeat the full frequency sweep
tmin  = 0.0    # Start time for the frequency sweep [s]
tres  = 0.05   # Temporal resolution [s] <-- Sets number of tones to use in a local frequency sweep
rate  = 44100  # Audio sampling rate [Hz]
Lmin  = 22.0   # Minimum sound pressure level [dB]
Lmax  = 56.0   # Maximum sound pressure level [dB]
inst = risset(H0=H0, f0=f0, N5th=N5th, Nrep=Nrep, tmin=tmin, tres=tres, rate=rate, Lmin=Lmin, Lmax=Lmax)

# Create the Risset glissando
mmin  = -25  # Minimum m-value
mmax  = 25   # Maximum m-value
chans = 1    # Number of audio channels (1 for mono, 2 for stereo)
sampwidth = 2  # Sample width (needed for writing out the WAV file)
fwav = "/Users/salvesen/outreach/asom/expansion/results/audio/rising-" + tag + ".wav"
fh5  = "/Users/salvesen/outreach/asom/expansion/results/audio/rising-" + tag + ".h5"
#inst.glissando(mmin=mmin, mmax=mmax, chans=chans, sampwidth=sampwidth, fwav=fwav, fh5=fh5)

'''
# Read in the times/chunks data created in the line above
f = h5py.File(fh5, 'r')
alltimes = f['alltimes'][:]
chunks   = f['allchunks'][:]
freqs    = f['freqs'][:,:]
Aenvl    = f['Aenvl'][:,:]
times    = f['times'][:]
f.close()
'''

# Plot the frequency comb
#froot = "/Users/salvesen/outreach/asom/expansion/results/frames/freqcomb-" + tag
#inst.plot_freqcomb(freqs=freqs, Aenvl=Aenvl, froot=froot)
#>> ffmpeg -f image2 -framerate 19.9992443502 -i 'freqcomb_%04d.png' -s 1260X1260 -pix_fmt yuv420p freqcomb.mp4
#>> ffmpeg -i freqcomb.mp4 -i rising.wav -c:v copy -c:a aac -strict experimental -shortest rising-freqcomb.mp4

# Plot the spectrogram
#froot = "/Users/salvesen/outreach/asom/expansion/results/frames/spectrogram-" + tag
#inst.plot_spectrogram(times=times, chunks=chunks, froot=froot, nperseg=2**15)
#>> ffmpeg -f image2 -framerate 19.9992443502 -i 'spectrogram_%04d.png' -s 1260X1260 -pix_fmt yuv420p spectrogram.mp4

# Plot the waveform
fout = "/Users/salvesen/outreach/asom/expansion/results/waveform.png"
#inst.plot_waveform(times=alltimes, chunks=chunks, fout=fout)


#====================================================================================================
# GENERATE THE VIDEO FRAMES

# Frame specifications
fps    = 12                  # Frame rate [frames/second]
Prot   = 4 * 60.0            # Rotation period [seconds]
dtheta = 360.0 / Prot / fps  # [degrees/frame]

print ""
print "Is dtheta a round number? ", dtheta
print ""

# Hubble time
km2cm  = 1.0e5                          # [cm km^-1]
Mpc2pc = 1.0e6                          # [pc Mpc^-1]
pc2cm  = 3.08567758149137e18            # [cm pc^-1]
H0cgs  = H0 * km2cm / (Mpc2pc * pc2cm)  # Hubble constant in CGS units [s^-1]
tHcgs  = 1.0 / H0cgs                    # Hubble time in CGS units [s]
sec2yr = 1.0 / (3600 * 24 * 365.25)     # [yr s^-1]
yr2Gyr = 1.0 / 1.0e9                    # [Gyr yr^-1]
tHGyr  = tHcgs * sec2yr * yr2Gyr        # Hubble time [Gyr]
tH     = tHGyr * 1.0e9                  # Hubble time [yr]

# Time and Size arrays
tmin  = 0                                  # Start time [Gyr]
tmax  = 43200                              # End time [Gyr] <-- 12 hours = 43,200 seconds
Nf    = int(fps * (tmax - tmin) + 1)       # Number of frames
t_arr = np.linspace(tmin, tmax, Nf) * 1e9  # [yr]
a_arr = []
for i in np.arange(Nf):
    pow = t_arr[i] / tH
    if (pow < 709):
        a_arr.append(np.exp(pow))
    else:
        a_arr.append(Decimal(np.exp(1))**int(np.floor(pow)))

Npix   = 2048  # Number of pixels for the CMB rendering
pixres = 1080  # Number of y-pixels for a frame (high resolution = 1080p)

# Generate the CMB renderings
#cmb(t_arr=t_arr, Npix=Npix, dtheta=dtheta, pixres=pixres)

# Generate the frames
frames(t_arr=t_arr, a_arr=a_arr, dtheta=dtheta, pixres=pixres)


