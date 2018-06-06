import numpy as np
import pylab as plt
import pyaudio
import wave
import struct
import h5py
from scipy.signal import spectrogram, correlate
from subprocess import call
from matplotlib.ticker import NullFormatter, MultipleLocator, ScalarFormatter
from matplotlib import cm, colors

#====================================================================================================
class risset():

    '''
    Purpose:
    --------
    Sonify the accelerated expansion of the Universe with a Risset glissando (D. W. Kammler 2000, pp. 725-727).

    Notes:
    ------
    Add flanger or reverb to make it sound creepier? --> http://manual.audacityteam.org/man/reverb.html
    http://alien.slackbook.org/blog/fixing-audio-sync-with-ffmpeg/
    
    I could not perfectly match the waveform at interfaces where one full frequency sweep ends and the next one begins.
    This leads to a noticeable popping sound, but I have come to accept it as a minor imperfection.
    '''

    #====================================================================================================
    def __init__(self, H0=70.49, f0=16.0, N5th=192, Nrep=550, tmin=0.0, tres=0.05, rate=44100, Lmin=22.0, Lmax=56.0):
        '''
        Inputs:
        -------
        H0   - Hubble constant [km s^-1 Mpc^-1]
        f0   - "Dimensionless frequency" @ t = 0 <-- f0 is "really" (f0 * tH)
        N5th - Number of musical 5th intervals to ascend over the full frequency sweep
        Nrep - Number of times to repeat the full frequency sweep
        tmin - Start time for the frequency sweep [s]
        tres - Temporal resolution [s] <-- Sets number of tones to use in a local frequency sweep
        rate - Audio sampling rate [Hz]
        Lmin - Minimum sound pressure level [dB]
        Lmax - Maximum sound pressure level [dB]
        '''
        # Choose the "dimensionless frequency": f0 is "really" (f0 * tH)
        self.f0 = f0  # This is our starting frequency [-] @ t = 0

        # Audio inputs
        self.tmin  = tmin   # Start time for the frequency sweep [s]
        self.tres  = tres   # Temporal resolution [s] <-- Sets number of tones to use in a local frequency sweep
        self.rate  = rate   # Audio sampling rate [Hz]
        self.Lmin  = Lmin   # Minimum sound pressure level [dB]
        self.Lmax  = Lmax   # Maximum sound pressure level [dB]
        self.Nrep  = Nrep   # Number of times to repeat the full frequency sweep

        # Unit conversions
        km2cm   = 1.0e5                       # [cm km^-1]
        Mpc2pc  = 1.0e6                       # [pc Mpc^-1]
        pc2cm   = 3.08567758149137e18         # [cm pc^-1]
        sec2yr  = 1.0 / (3600 * 24 * 365.25)  # [yr s^-1]
        yr2Gyr  = 1.0 / 1.0e9                 # [Gyr yr^-1]

        # Cosmology inputs
        H0cgs = H0 * km2cm / (Mpc2pc * pc2cm)  # Hubble constant in CGS units [s^-1]
        tHcgs = 1.0 / H0cgs                    # Hubble time in CGS units [s]
        tHGyr = tHcgs * sec2yr * yr2Gyr        # Hubble time [Gyr]

        # Let's work in [Gyr] time units (each passing second in the sonification will correspond to 1 Gyr)
        self.tH    = tHGyr          # [s]
        self.alpha = 1.0 / self.tH  # [s^-1]

        # We want the sonification to sound "harmonious," so we use intervals of musical fifths
        self.N5th  = N5th                    # Number of musical 5th intervals to ascend over the full frequency sweep
        i5th       = 3.0 / 2.0               # Interval of a musical fifth (factor of 3:2)
        self.dt5th = np.log(i5th) * self.tH  # Time interval to increase the local frequency by a musical fifth [s]

        # Number of musical fifth intervals to use, which determines: dt = (tmax-tmin) and the range [fmin,fmax]
        self.tmax = np.log(i5th * self.N5th) / self.alpha  # End time for the frequency sweep [s]
        self.dt   = self.tmax - self.tmin                  # Time interval for the frequency sweep [s]
        self.tmid = (self.tmin + self.tmax) * 0.5          # Midpoint time [s]

        # Print out some information
        print "\nFor 12 hours of audio, the snippet must repeat this many times: ", (12 * 3600.0) / self.tmax
        print "\nTime for a musical fifth [seconds]: ", self.dt5th
        Nt = int(np.floor((self.tmax - self.tmin) / self.tres))
        print "\nFrames per second for movies using coarse time grid: ", Nt / self.dt


    #====================================================================================================
    # HELPER FUNCTIONS:

    # Compute the local frequency [Hz] as a function of time [s]
    def freq(self, t):
        f = self.f0 * np.exp(self.alpha * t)  # [Hz]
        return f

    # Compute the Gaussian amplitude envelope [dB] as a function of time [s]
    def amp_envl(self, t):
        lnLrat = np.log(self.Lmin / self.Lmax)
        exparg = (t - self.tmid) / (0.5 * self.dt)
        AdB    = self.Lmax * (np.exp(exparg**2))**lnLrat  # [dB]
        Arms   = 10.0**(AdB / 20.0)                  # [rms]
        return Arms

    #----------------------------------------------------------------------------------------------------
    # Determine the min/max m-values accessible to the range of human hearing [fmin,fmax] = [20 Hz, 20 kHz]
    def mlims(self, fmin=20, fmax=20e3):

        # Compute the minimum m-value to use in the summation for w(t)
        mmin   = 0
        f_mmin = self.freq(t=(self.tmax + mmin * self.dt5th))  # [Hz]
        while (f_mmin >= fmin):
            mmin   = mmin - 1
            f_mmin = self.freq(t=(self.tmax + mmin * self.dt5th))  # [Hz]

        # Compute the maximum m-value to use in the summation for w(t)
        mmax   = 0
        f_mmax = self.freq(t=(self.tmin + mmax * self.dt5th))  # [Hz]
        while (f_mmax <= fmax):
            mmax   = mmax + 1
            f_mmax = self.freq(t=(self.tmin + mmax * self.dt5th))  # [Hz]

        return mmin, mmax
    #----------------------------------------------------------------------------------------------------

    #====================================================================================================
    # Create the Risset glissando!
    def glissando(self, mmin=-25, mmax=25, chans=1, sampwidth=2, \
                  fwav="/Users/salvesen/outreach/asom/expansion/results/audio/risset.wav", \
                  fh5="/Users/salvesen/outreach/asom/expansion/results/audio/risset.h5"):
        '''
        Inputs:
        -------
        mmin,mmax - Min/Max m-values
        chans     - Number of audio channels (1 for mono, 2 for stereo)
        sampwidth - Sample width (needed for writing out the WAV file)
        fwav      - Output WAV file
        fh5       - Output HDF5 file
        '''
        
        #----------------------------------------------------------------------------------------------------
        # GRIDS:

        # Array of m-values to loop over
        #mmin, mmax = self.mlims() <-- Does not set a wide enough range, set by hand instead in the inputs
        mvals = np.arange(mmin, mmax+1)
        Nm    = len(mvals)

        # Nt    - Number of time intervals to use for the full frequency sweep
        # times - Array of "coarse" times [s] <-- clip the final time bin
        # dtchk - Duration of a chunk/"chord" [s] <-- will be slightly different from "tres" input
        Nt    = int(np.floor((self.tmax - self.tmin) / self.tres))
        times = np.linspace(self.tmin, self.tmax, Nt+1)[0:-1]
        dtchk = (self.tmax - self.tmin) / Nt

        # Array of "fine" times [s]
        Ntsub    = int(np.floor(self.rate * dtchk))      # Number of time intervals contained in a chunk [s]
        subtimes = np.linspace(0, dtchk, Ntsub+1)[0:-1]  # Array of "fine" times [s] <-- clip the final time bin
        dtsub    = dtchk / Ntsub                         # Duration of a sub-time interval [s]

        # Array of all times [s]: Go from tmin --> tmax with dtsub spacing
        alltimes = []
        for i in np.arange(Nt):
            alltimes.append(times[i] + subtimes)
        alltimes = np.concatenate(alltimes)
        #----------------------------------------------------------------------------------------------------

        # Collect arrays of frequencies and amplitude envelopes on the "coarse" time grid
        freqs = np.zeros([Nt, Nm, self.Nrep])
        Aenvl = np.zeros([Nt, Nm, self.Nrep])
        for k in np.arange(self.Nrep):
            for i in np.arange(Nt):
                for j in np.arange(Nm):
                    tnow = times[i] + mvals[j] * self.dt5th  # [s]
                    freqs[i,j,k] = self.freq(t=tnow)         # [-]
                    Aenvl[i,j,k] = self.amp_envl(t=tnow)     # [rms]
        
                # Check that the sound-pressure envelope is constant across all tones at any given time
                const = np.sum(Aenvl[0,:,0])  # We want: sum_{m=mmin}^{m=mmax} A(t,m) = constant
                rdiff = np.abs((np.sum(Aenvl[i,:,k]) - const) / const)  # Relative difference
                if (rdiff > 1e-3):
                    print "\nWARNING: The sound-pressure level envelope is not constant across all tones."
                    print "           Try expanding the current range in mmin/mmin: ", mmin, mmax


        # Print out some information
        imid = int(Nt * 0.5)
        jmid = int((mmax-mmin) * 0.5)
        print ""
        print "STARTING FREQUENCY [Hz]: ", freqs[0,jmid,0]
        print "MIDPOINT FREQUENCY [Hz]: ", freqs[imid,jmid,0]
        print "ENDING   FREQUENCY [Hz]: ", freqs[-1,jmid,0]
        print ""
        print "STARTING AMPLITUDE: ", Aenvl[0,jmid,0]
        print "MIDPOINT AMPLITUDE: ", Aenvl[imid,jmid,0]
        print "ENDING   AMPLITUDE: ", Aenvl[-1,jmid,0]
        print ""
        iroll = 0
        # Initialize the audio chunks list
        allchunks = []

        # Initialize the phase array
        phase = np.zeros(Nm)

        # Initialize the amplitude offset (this is how I fix the "popping" when repeating the full sweep)
        offset  = 0.0
        endfreq = self.freq(t=self.tmax)

        # Loop over the "coarse" time grid
        print "\nCreating audio chunks..."
        icnt = 1
        for k in np.arange(self.Nrep):
    
            for i in np.arange(Nt):
        
                # Initialize the audio chunk for the current sub-time interval
                chunk = np.zeros(Ntsub)

                # Loop over notes to construct a "chord" on the current "fine" time grid
                for j in np.arange(Nm):
        
                    # Choose the phase that matches the end of the sine wave from the previous tone
                    # (Need to track the phase from the previous tone for each octave being sounded)
                    if (i > 0):
                        dphase = 2.0 * np.pi * freqs[i-1,j,k] * Ntsub * dtsub
                        # Require the phases to be bounded by [0, 2 pi]
                        N2pi = np.floor(dphase / (2.0 * np.pi))
                        if (dphase >= (2.0 * np.pi)): dphase = dphase - (2.0 * np.pi) * N2pi
                        if (dphase < 0.0): dphase = dphase + (2.0 * np.pi) * N2pi
                        phase[j] = phase[j] + dphase
                        while(phase[j] >= 2.0*np.pi): phase[j] = phase[j] - 2.0 * np.pi
                        while(phase[j] < 0.0): phase[j] = phase[j] + 2.0 * np.pi
                
                    # Calculate the sine wave with the Gaussian amplitude envelope applied
                    w = Aenvl[i,j,k] * np.sin(2.0 * np.pi * freqs[i,j,k] * np.arange(Ntsub) * dtsub + phase[j])

                    # Create the audio chunk/"chord" by summing up the individual tones
                    chunk = chunk + w
    
                # Account for the offset at the interface between the end/start of a full frequency sweep
                chunk = chunk + offset
                
                # Add this chunk/"chord" to the list
                allchunks.append(chunk)

                # Give a progress report every so often
                if ((icnt % 100) == 0):
                    pctComplete = float(icnt) / float(Nt) * 100 / float(self.Nrep)
                    print "    % Complete: ", '{:.1f}'.format(pctComplete)
                icnt = icnt + 1


            #----------------------------------------------------------------------------------------------------
            # DOING MY BEST TO MATCH THE WAVEFORMS AT END/START FREQUENCY SWEEP INTERFACES
            
            # Calculate the phase for the new frequency sweep that is about to start
            for j in np.arange(Nm):
                dphase = 2.0 * np.pi * (freqs[i,j,k] - endfreq) * self.tmax  # <-- I'm still not sure if this is right.
                # Require the phases to be bounded by [0, 2 pi]
                N2pi = np.floor(dphase / (2.0 * np.pi))
                if (dphase >= (2.0 * np.pi)): dphase = dphase - (2.0 * np.pi) * N2pi
                if (dphase < 0.0): dphase = dphase + (2.0 * np.pi) * N2pi
                phase[j] = phase[j] + dphase
                while(phase[j] >= 2.0*np.pi): phase[j] = phase[j] - 2.0 * np.pi
                while(phase[j] < 0.0): phase[j] = phase[j] + 2.0 * np.pi


            # Match the amplitudes across the interface of an old/new frequency sweep
            if (k < (self.Nrep-1)):

                # Amplitude of the end point of the chunk (end of last frequency sweep)
                endchunk  = allchunks[-1][-1]
                penchunk  = allchunks[-1][-2]    # Penultimate chunk!
                diffchunk = endchunk - penchunk  # This allow us to match the gradient at the tail of the waveform

                # Calculate the amplitude of the start point of the next chunk (next frequency sweep)
                nextchunk = 0.0
                for j in np.arange(Nm):
                    w_next    = Aenvl[0,j,k+1] * np.sin(phase[j])
                    nextchunk = nextchunk + w_next
                
                # Calculate the offset between the end of the old chunk and start of the new chunk
                offset = endchunk - nextchunk + diffchunk
            #----------------------------------------------------------------------------------------------------


        # Concatenate the list of chunks into one array
        allchunks = np.concatenate(allchunks)

        # Repeat the sequence Ncycles times
        alltimeslist = []
        for i in np.arange(self.Nrep):
            alltimeslist.append(alltimes + i * (self.tmax - self.tmin))
        alltimes = np.concatenate(alltimeslist)

        '''
        # Play the audio chunks list using PyAudio
        p = pyaudio.PyAudio()
        format = pyaudio.paFloat32
        stream = p.open(format=format, channels=chans, rate=self.rate, output=1)
        stream.write(allchunks.astype(np.float32).tostring())
        stream.close()
        p.terminate()
        '''
        
        # Write the audio stream to a WAV file
        print "\nWriting out the WAV file..."
        volmax =  2**(8 * sampwidth - 1) - 1
        volmin = -2**(8 * sampwidth - 1)
        allchunks = (allchunks / np.max(np.abs(allchunks))) * volmax  # Normalize the amplitudes
        
        wf = wave.open(fwav, 'wb')
        wf.setnchannels(chans)
        wf.setsampwidth(sampwidth)
        wf.setframerate(self.rate)
        Nallchunks = len(allchunks)
        for i in np.arange(Nallchunks):
            data = struct.pack('<h', allchunks[i])
            wf.writeframesraw(data)
        
        wf.writeframes('')
        wf.close()
        
        
        # Write the audio stream to an HDF5 file
        print "\nWriting out the HDF5 file..."
        f = h5py.File(fh5, 'w')
        f.create_dataset('alltimes', data=alltimes)
        f.create_dataset('allchunks', data=allchunks)
        f.create_dataset('freqs', data=freqs[:,:,0])  # Just keep the first frequency sweep
        f.create_dataset('Aenvl', data=Aenvl[:,:,0])  # Just keep the first frequency sweep
        f.create_dataset('times', data=times)         # "Coarse" times
        f.close

        print "\nFinished making the audio file!\n"

    #====================================================================================================
    # PLOTTING FUNCTIONS
    
    #----------------------------------------------------------------------------------------------------
    def plot_freqcomb(self, freqs, Aenvl, froot):
        
        # Plotting preferences
        dpi   = 300
        lw    = 1.5
        fs    = 18
        ls    = 10
        pad   = 5
        tlmaj = 5.0
        tlmin = 2.5
        xsize, ysize = 4.2, 4.2
        left, right, bottom, top = 0.175, 0.975, 0.175, 0.975

        # Normalize the sound pressure envelope
        Aenvl = Aenvl / np.max(Aenvl)
        
        # Number of frequencies played at any given time
        Nm   = len(freqs[0,:])
        jmid = int(Nm / 2)
        
        # Plot the frequency comb
        xmin = self.f0
        xmax = self.f0 * (3.0/2.0) * self.N5th
        ymin = 0.0
        ymax = 1.05
        yinc_maj, yinc_min = 0.1, 0.05
        ymajorLocator = MultipleLocator(yinc_maj)
        yminorLocator = MultipleLocator(yinc_min)
        plt.rcParams['axes.linewidth'] = lw

        # Setup the base plot
        fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        ax = fig.add_subplot(111)
        ax.set_xlabel(r"${\rm Frequency\ (Hz)}$", fontsize=fs, labelpad=pad)
        ax.set_ylabel(r"${\rm Volume\ (Normalized)}$", fontsize=fs, labelpad=pad)
        ax.set_xscale('log')
        ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
        ax.tick_params('both', direction='in', labelsize=ls, length=tlmaj, width=lw, which='major', pad=pad)
        ax.tick_params('both', direction='in', labelsize=ls, length=tlmin, width=lw, which='minor')

        # Plot the sound pressure envelope
        ax.plot(freqs[:,jmid], Aenvl[:,jmid], 'b-', linewidth=lw, zorder=3)

        # Loop through each time
        Nt   = len(freqs[:,0])
        Nz   = np.ceil(np.log10(Nt)).astype(int)
        icnt = 1
        for t in np.arange(Nt):
            
            # Loop through each frequency, plot the comb
            comb = []
            for m in np.arange(Nm):
                line = ax.plot([freqs[t,m],freqs[t,m]], [0,Aenvl[t,m]], 'r-', linewidth=lw)
                comb.append(line)
            
            # Save the figure with the current frequency comb
            fout = froot + "_" + str(t).zfill(Nz) + ".png"
            fig.savefig(fout, bbox_inches=0, dpi=dpi)
            
            # Remove the frequency comb line-by-line
            for m in np.arange(Nm):
                line = comb[m]
                line.pop(0).remove()
            
            # Give a progress report every so often
            if ((icnt % 10) == 0):
                pctComplete = float(icnt) / float(Nt) * 100 / float(self.Nrep)
                print "    % Complete: ", '{:.1f}'.format(pctComplete)
            icnt = icnt + 1
        
        # Close the plotting environment
        plt.close()
        
        # Open the last plot made
        call("open " + fout, shell=True)

    
    #----------------------------------------------------------------------------------------------------
    def plot_spectrogram(self, times, chunks, froot, nperseg=2**15):

        # Plotting preferences
        dpi   = 300
        lw    = 1.5
        fs    = 18
        ls    = 10
        pad   = 5
        tlmaj = 5.0
        tlmin = 2.5
        xsize, ysize = 4.2, 4.2
        left, right, bottom, top = 0.175, 0.975, 0.175, 0.975

        # Spectrogram
        f, t, Sxx = spectrogram(x=chunks, fs=self.rate, window=('tukey',0.25), nperseg=nperseg)

        # Plot the spectrogram
        xmin = self.f0
        xmax = self.f0 * (3.0/2.0) * self.N5th
        ymin = self.tmin
        ymax = self.tmax
        yinc_maj, yinc_min = 10, 5
        ymajorLocator = MultipleLocator(yinc_maj)
        yminorLocator = MultipleLocator(yinc_min)
        plt.rcParams['axes.linewidth'] = lw
        
        # Setup the base plot
        fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        ax = fig.add_subplot(111)
        ax.set_xlabel(r"${\rm Frequency\ (Hz)}$", fontsize=fs, labelpad=pad)
        ax.set_ylabel(r"${\rm Time\ (seconds)}$", fontsize=fs, labelpad=pad)
        ax.set_xscale('log')
        ax.set_xticks([20, 50, 100, 200, 500, 1000, 2000])
        ax.xaxis.set_major_formatter(ScalarFormatter())
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.yaxis.set_major_locator(ymajorLocator)
        ax.yaxis.set_minor_locator(yminorLocator)
        ax.tick_params('both', direction='in', labelsize=ls, length=tlmaj, width=lw, which='major', pad=pad)
        ax.tick_params('both', direction='in', labelsize=ls, length=tlmin, width=lw, which='minor')

        # Plot the spectrogram
        vmin = 1.0#np.min(Sxx)*1e6
        vmax = np.max(Sxx)
        ax.pcolormesh(f, t, Sxx.T, cmap='Greys', norm=colors.LogNorm(vmin=vmin, vmax=vmax))  # <-- Note: We had to transpose the array!

        # Loop through each time
        Nt   = len(times)
        Nz   = np.ceil(np.log10(Nt)).astype(int)
        icnt = 1
        for t in np.arange(Nt):

            # Plot a line marking the current time
            line = ax.plot([xmin,xmax], [times[t],times[t]], 'r-', linewidth=lw)
        
            # Save the figure with the current time marked
            fout = froot + "_" + str(t).zfill(Nz) + ".png"
            fig.savefig(fout, bbox_inches=0, dpi=dpi)

            # Remove the line marking the current time
            line.pop(0).remove()

            # Give a progress report every so often
            if ((icnt % 10) == 0):
                pctComplete = float(icnt) / float(Nt) * 100 / float(self.Nrep)
                print "    % Complete: ", '{:.1f}'.format(pctComplete)
            icnt = icnt + 1

        # Close the plotting environment
        plt.close()

        # Open the last plot made
        call("open " + fout, shell=True)


    #----------------------------------------------------------------------------------------------------
    def plot_waveform(self, times, chunks, fout):
    
        # Plotting preferences
        dpi   = 300
        lw    = 1.5
        fs    = 18
        ls    = 10
        pad   = 5
        tlmaj = 5.0
        tlmin = 2.5
        xsize, ysize = 4.2, 4.2
        left, right, bottom, top = 0.175, 0.975, 0.175, 0.975

        # Normalize the amplitude
        chunks = chunks / np.max(chunks)

        # Useful stuff
        Nt    = int(np.floor((self.tmax - self.tmin) / self.tres))
        dtchk = (self.tmax - self.tmin) / Nt
        xmin  = np.min(times)
        xmax  = np.max(times)
        ymin  = np.min(chunks)
        ymax  = np.max(chunks)

        # Plot the waveform
        plt.rcParams['axes.linewidth'] = lw
        fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
        fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
        ax = fig.add_subplot(111)
        ax.set_xlabel(r"${\rm Time\ (seconds)}$", fontsize=fs, labelpad=pad)
        ax.set_ylabel(r"${\rm Amplitude\ (Normalized)}$", fontsize=fs, labelpad=pad)
        ax.tick_params('both', direction='in', labelsize=ls, length=tlmaj, width=lw, which='major', pad=pad)
        ax.tick_params('both', direction='in', labelsize=ls, length=tlmin, width=lw, which='minor')
        ax.set_xlim((self.Nrep-1)*self.tmax-0.5*dtchk, (self.Nrep-1)*self.tmax+0.5*dtchk)
        ax.set_ylim(-1.0, 1.0)
        ax.plot(times, chunks, 'ko-', linewidth=0.1, markersize=1)
        ax.plot([(self.Nrep-1)*self.tmax,(self.Nrep-1)*self.tmax],[ymin,ymax], 'r--', linewidth=0.1)
        fig.savefig(fout, bbox_inches=0, dpi=dpi)
        plt.close()
        
        call("open " + fout, shell=True)
