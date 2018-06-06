import numpy as np
import pylab as plt
import pyaudio

'''
Create a Shepard's tone!

Notes:
------
To install pyaudio:
>> brew install portaudio
>> pip install pyaudio

Check out pydub?

A Gaussian envelope might be a better choice.
If we do this using MIDIUtil, do we get this annoying blippy behavior?
To get rid of the popping, I need to match the phases of the sine waves between successive tones.
https://stackoverflow.com/questions/36438850/how-to-remove-pops-from-concatented-sound-data-in-pyaudio

Check if the sine waves match up with each other smoothly for successive tones!
'''
# Some inputs
fmin = 4.863  # Minimum frequency [Hz] (cps = cycles per second = Hz)
tmax = 12 #640     # Number of notes to use within an octave
cmax = 10 #8     # Number of octaves to use in the spectrum
Lmin = 22.0   # Minimum volume [dB]
Lmax = 56.0   # Maximum volume [dB]

rate     = 44100 #8000 # Audio sampling rate [Hz]
duration = 0.12 #0.125   # Duration of each tone [sec]
tonegap  = 0.12 #0.0   # Gap between tones [sec]
mono     = 1   # mono = 1 channel
stereo   = 2   # stereo = 2 channels
Ncycles  = 4#2  # Number of times to repeat the frequency sweep

#====================================================================================================
# SHEPARD TONE FREQUENCIES AND VOLUMES

# Initialize the frequency and sound-pressure level arrays
f = np.zeros([tmax, cmax])
L = np.zeros([tmax, cmax])

# Loop over tones
for t in np.arange(tmax):

    # Loop over octaves
    for c in np.arange(cmax):

        # Calculate the frequency [Hz]
        power  = (float(c) * float(tmax) + float(t)) / float(tmax)
        f[t,c] = fmin * 2.0**power
        
        # Calculate the sound pressure level [dB]
        # (adopt a cosine envelope for now)
        theta  = 2.0 * np.pi * power / cmax
        L[t,c] = Lmin + (Lmax - Lmin) * (1.0 - np.cos(theta)) / 2.0

    # Check that Sum_{c=1}^{cmax} L(t,c) = constant
    const = np.sum(L[0,:])
    diff  = np.sum(L[t,:]) - const
    if (np.abs(diff) > 1e-6):
        print "\nWARNING: The sound-pressure level envelope is not constant across all tones.\n"

# Calculate the (continuous) sound envelope
theta = np.linspace(0, 2.0*np.pi, 1000)
fenv  = fmin * 2.0**(theta * cmax / (2.0 * np.pi))
Lenv  = Lmin + (Lmax - Lmin) * (1.0 - np.cos(theta)) / 2.0


#====================================================================================================
# CREATE AUDIO CHUNKS FOR THE SHEPARD TONE

def sine(frequency, factor, duration=0.12, rate=44100):
    '''
    Generate a sine wave array that can be used as an audio chunk.

    Inputs:
    -------
    frequency - Frequency of the sine wave tone to be produced [Hz]
    duration  - Duration of the sine wave tone to be produced [sec]
    rate      - Sampling rate for the audio signal [Hz]
    
    Output:
    -------
    Sine wave array for use as input as an audio chunk.
    
    Notes:
    ------
    A sampling rate of 44.1 kHz gives a 22 kHz maximum frequency.
    This is about the highest that humans can hear.
    '''
    length   = int(duration * rate)
    factor   = float(frequency) * (2.0 * np.pi) / rate
    sinewave = np.sin(np.arange(length) * factor)
    return sinewave


# Initialize the audio chunks list
chunks = []

# Chunk of silence (to be used inbetween two sound chunks)
chunkgap = sine(frequency=0, duration=tonegap, rate=rate)

# Loop over tones
for t in np.arange(tmax):

    # Initialize the audio chunk
    chunk = sine(frequency=0.0, duration=duration, rate=rate)

    # Loop over octaves
    for c in np.arange(cmax):
        
        # Create the tone at the current octave
        dB  = L[t,c]
        rms = np.log10(dB / 20.0)
        sinewave = sine(frequency=f[t,c], duration=duration, rate=rate)
        tone     = sinewave * rms

        # Create the audio chunk by combining the individual tones
        chunk = chunk + tone


    # Add this chunk to the list and tag on a moment of silence afterwards
    chunks.append(chunk)
    chunks.append(chunkgap)

# Repeat the sequence Ncycles times (use extend, not append)
chunks.extend(chunks * int(Ncycles - 1))

# Concatenate the list of chunks into one array
allchunks = np.concatenate(chunks)

# Play the audio chunks list using PyAudio
p = pyaudio.PyAudio()
stream = p.open(format=pyaudio.paFloat32, channels=mono, rate=rate, output=1)
stream.write(allchunks.astype(np.float32).tostring())
stream.close()
p.terminate()

# Volume, use audioop?
#data = stream.read(CHUNK)
#rms = audioop.rms(data,2)
#decibel = 20 * np.log10(rms)


#====================================================================================================
# PLOT THE AUDIO SPECTRA

fout = "shepardtone.png"

dpi = 300
lw = 2
fs = 30
ls = 20
pad = 12
xmin = fmin - 2.0
xmax = 2.0 * np.max(f)
ymin = Lmin - 2.0
ymax = Lmax + 2.0
tlmaj = 10.0
tlmin = 5.0
xsize, ysize = 8.4, 8.4
left, right, bottom, top = 0.15, 0.95, 0.15, 0.95
plt.rcParams['axes.linewidth'] = lw

fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
ax = fig.add_subplot(111)
ax.set_xlabel(r"${\rm Frequency}$", fontsize=fs, labelpad=pad)
ax.set_ylabel(r"${\rm Sound\ Pressure\ Level\ (dB)}$", fontsize=fs, labelpad=pad)
ax.set_xscale('log')
ax.set_xlim(xmin, xmax)
ax.set_ylim(ymin, ymax)
ax.tick_params('both', direction='in', labelsize=ls, length=tlmaj, width=lw, which='major', pad=pad)
ax.tick_params('both', direction='in', labelsize=ls, length=tlmin, width=lw, which='minor')

t = 0
for c in np.arange(cmax):
    ax.plot([f[t,c],f[t,c]], [0,L[t,c]], 'k-', linewidth=lw)

ax.plot(fenv, Lenv, 'k--', linewidth=lw)

fig.savefig(fout, bbox_inches=0, dpi=dpi)
plt.close()


#====================================================================================================
# PLOT THE SINE WAVES


