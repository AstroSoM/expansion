'''
# Some crap to get mayavi to work in a hacky way
import matplotlib
matplotlib.use('WXAgg')
matplotlib.interactive(False)
from mayavi import mlab
'''
import numpy as np
import healpy as hp
import h5py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib import cm, colors
from matplotlib.patches import FancyArrowPatch
from bignumber import *
from matplotlib import image
from decimal import Decimal

'''
Purpose:
--------
Plot the frames needed to make the video of the CMB on a sphere, accompanied by the evolution of time and the scale factor.

Notes:
------
Try outputting each CMB image separately. Make a move of the full rotation and inspect it by eye.

Then overlay two images on top of each other!
https://stackoverflow.com/questions/10640114/overlay-two-same-sized-images-in-python
'''



#====================================================================================================
# For YouTube production, determine the appropriate (xsize,ysize) for the figures (16:9 aspect)
def XYsize(pixres, dpi):
    '''
    Inputs:
    -------
    pixres - Resolution (240, 360, 720, 1080, etc.)
    dpi    - Dots per inch
    '''
    aspect = 16.0 / 9.0
    Nxpix  = float(pixres) * aspect
    Nypix  = float(pixres)
    xsize  = Nxpix / dpi
    ysize  = Nypix / dpi
    return xsize, ysize


#====================================================================================================

# Do time in billions of years
def timestr(t):
    tdiv = t / 1e9
    tint = np.floor(tdiv).astype(int)
    ttxt = r"$\mathregular{" + "{:,}".format(tint) + "}$" + "\n" + "billion" + "\n" + "years into" + "\n" + "the future."
    return ttxt


def format_e(n):
    x = '%E' % n
    if (x == 'INF'): x = str(n)
    num = np.round(float(x.split('E')[0])).astype(int)  # <-- Could floor instead of round, but w/e
    pow = np.round(float(x.split('E')[1])).astype(int)  # <-- Could floor instead of round, but w/e
    #return r"$\mathregular{" + str(num) + "}$" + r"$\times$" + r"$\mathregular{10^{" + "{:,}".format(pow) + "}}$"
    return r"$\mathregular{10^{" + "{:,}".format(pow) + "}}$"

def round_sig(x, sig=3):
    return np.round(x, sig-int(np.floor(np.log10(abs(x))))-1)

def sizestr(a):
    name, power = bignumber(a)
    if isinstance(a, float):
        adiv = a / 10.0**power
    else:
        adiv = a / Decimal(10)**int(power)

    abig = 1e9
    if (adiv < abig):
        aint = np.floor(round_sig(adiv, 3)).astype(int)  # <-- Using floor is best here early on when numbers are not huge yet
        atxt = r"$\mathregular{" + "{:,}".format(aint) + "}$" + "\n" + name + "\n" + "times bigger" + "\n" + "than it is today."
    else:
        astr = format_e(adiv)
        atxt = astr + "\n" + name + "\n" + "times bigger" + "\n" + "than it is today."
    return atxt


#====================================================================================================
def cmb(t_arr, Npix=32, dtheta=1.0, pixres=1080, ffits="../data/COM_CMB_IQU-commander-field-Int_2048_R2.01_full.fits", \
        rroot="/Users/salvesen/outreach/asom/expansion/results/frames/cmb"):
    '''
    Purpose:
    --------
    Render the CMB sphere for every azimuthal angle and save the outputs.
    (Something went wrong if the output files are not numbered 1,2,3,...)
    
    Inputs:
    -------
    t_arr  - Array of times [Gyr] <-- We really just need to know the size of this array, but w/e
    Npix   - Number of (phi, theta) pixels to use (2048 max)
    dtheta - Incremental rotation angle for the CMB sphere between frames [degrees/frame]
    pixres - Image resolution (240p, 360p, 480p, 720p, 1080p, etc.)
    ffits  - CMB map FITS file (https://pla.esac.esa.int/pla/)
    rroot  - Root name for the output rendered ('r') PNG files

    Notes:
    ------
    When choosing dtheta, make sure it divides nicely into 360 degrees...
    ...otherwise the rendering cannot use overlapping angles to its advantage as a trick to run the code faster

    I tried a simpler approach of rendering the CMB for a given angle and saving it with mlab.screenshot().
    This allows you to overlay the screenshot onto a matplotlib figure, but it was not rendering reliably.
    So now I render the CMB sphere for every angle upfront and save these so I can read them in later.
    '''
    #----------------------------------------------------------------------------------------------------

    # Load in the CMB map
    map   = hp.read_map(ffits)
    Nside = hp.npix2nside(len(map))

    # Construct the spherical grid
    Nphi   = Npix
    Ntheta = Npix
    phi    = np.linspace(0.0, 2.0*np.pi, Nphi)
    theta  = np.linspace(0.0, np.pi, Ntheta)

    # Project the map to a rectangular matrix Nx x Ny
    PHI, THETA = np.meshgrid(phi, theta)
    grid_pix   = hp.ang2pix(Nside, THETA, PHI)
    grid_map   = map[grid_pix]
    
    # Min/max map values for the colorbar (normalize the grid)
    print "\nMin/Max on grid [microK]:", np.ceil(np.min(grid_map)*1e6), np.ceil(np.max(grid_map)*1e6)
    vmin = -500e-6  # [microK]
    vmax = 500e-6   # [microK]
    norm = colors.Normalize(vmin=vmin, vmax=vmax, clip=False)
    
    # CMB colormap (https://zonca.github.io/2013/09/Planck-CMB-map-at-high-resolution.html)
    colombi1_cmap = colors.ListedColormap(np.loadtxt("Planck_Parchment_RGBA.txt"))
    colombi1_cmap.set_bad("gray")     # Color of missing pixels
    colombi1_cmap.set_under("white")  # Color of background, necessary if you want to use this colormap directly with hp.mollview(m, cmap=colombi1_cmap)
    cmap = colombi1_cmap.colors

    # Create a sphere
    r = 1.0
    x = r * np.sin(THETA) * np.cos(PHI)
    y = r * np.sin(THETA) * np.sin(PHI)
    z = r * np.cos(THETA)
    rmin = -1.0 * r
    rmax = r

    #----------------------------------------------------------------------------------------------------
    
    # Set the starting azimuth and elevation
    azim0 = 0.0   # [degrees]
    elev0 = 60.0  # [degrees]

    # We will loop through each time/size in a smart way...
    # ...output entire set of frames where the CMB sphere has a given azimuthal angle
    N         = len(t_arr)
    Nz        = np.ceil(np.log10(N)).astype(int)
    N_arr     = np.arange(N)
    azim_arr  = np.empty(N)
    rout_list = []
    for i in N_arr:
        
        # Calculate the azimuthal angle for frame i [0,360]
        azim = azim0 + i * dtheta
        while (azim >= 360.0): azim = azim - 360.0
        azim_arr[i] = azim

        # Calculate the output file name for rendering i
        rout_list.append(rroot + "_" + str(i).zfill(Nz) + ".png")

    # Sort the indexing array by azimuthal angle of the CMB sphere rendering
    # NOTE: kind='mergesort' does exactly what I want! :)
    isort  = np.argsort(azim_arr, kind='mergesort')
    N_sort = N_arr[isort]

    # Render the CMB sphere (using Mayavi) for every azimuthal angle
    print("\nRendering the CMB for each azimuthal angle...\n")
    Nxpix = pixres * (16.0 / 9.0)
    Nypix = pixres
    icnt  = 1
    dummy = -1.0 * dtheta
    azim_old = dummy
    for i in N_sort:
        
        # Check if the previous frame has a different azimuthal angle from the current frame
        # (Hacky if statement to replace "if (azim_now != azim_old)", which is not reliable)
        azim_now = azim_arr[i]
        if (np.abs(azim_now - azim_old) > 0.5*dtheta):
        #if ((np.abs(azim_now - azim_old) > 0.5*dtheta) and (i > 2706)):

            # Initialize the figure
            mlab.figure(0, bgcolor=(0,0,0), fgcolor=(0,0,0), size=(Nxpix, Nypix))  # Initialize the figure

            # Render the CMB sphere
            surf = mlab.mesh(x, y, z, extent=[-1,1,-1,1,-1,1], scalars=grid_map, vmin=vmin, vmax=vmax)  # Draw the CMB sphere
            surf.module_manager.scalar_lut_manager.lut.table = cmap     # Customize the colormap
            mlab.view(azimuth=azim_now, elevation=elev0, distance=6.7)  # Rotate the sphere

            # Save the figure and close
            rout = rout_list[i]
            mlab.savefig(rout, size=(Nxpix, Nypix))
            mlab.close()
        
        # Update azim_old
        azim_old = azim_now

        # Give a progress report every so often
        if ((icnt % 100) == 0):
            pctComplete = float(icnt) / float(N) * 100
            print "    % Complete: ", '{:.1f}'.format(pctComplete)
        icnt = icnt + 1


#====================================================================================================
def frames(t_arr, a_arr, dtheta=1.0, pixres=1080, \
           froot="/Users/salvesen/outreach/asom/expansion/results/frames/expansion", \
           rroot="/Users/salvesen/outreach/asom/expansion/results/frames/cmb"):
    '''
    Purpose:
    --------
    Plot every frame for the movie.
    
    Inputs:
    -------
    t_arr  - Array of times [Gyr]
    a_arr  - Array of scale factors/sizes [-]
    dtheta - Incremental rotation angle for the CMB sphere between frames [degrees/frame]
    pixres - Image resolution (240p, 360p, 480p, 720p, 1080p, etc.)
    froot  - Root name for the output frame PNG files
    rroot  - Root name for the output rendered ('r') PNG files

    '''
    # Plotting preferences
    fres = pixres / 1080.0
    dpi  = 200
    lw   = 2.0 * fres
    fsL  = 48 * fres
    fsS  = 24 * fres
    fsXS = 12 * fres
    mscl = 24 * fres
    left, right, bottom, top = 0.0, 1.0, 0.0, 1.0
    xsize, ysize = XYsize(pixres=pixres, dpi=dpi)
    plt.rcParams['axes.linewidth'] = lw
    plt.rcParams['axes.facecolor'] = 'k'
    plt.rcParams['axes.titlepad']  = -1.5*fsS

    # Fonts
    jsansR  = fm.FontProperties(fname='../fonts/JosefinSans-Regular.ttf')
    jsansB  = fm.FontProperties(fname='../fonts/JosefinSans-Bold.ttf')
    jsansSB = fm.FontProperties(fname='../fonts/JosefinSans-SemiBold.ttf')
    jsansBI = fm.FontProperties(fname='../fonts/JosefinSans-BoldItalic.ttf')

    # Initialize the figure and do various aesthetic things
    fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax = fig.add_subplot(111)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_title("The Universe", fontsize=fsL, fontproperties=jsansB, color='w')
    
    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])

    # Draw lines/vectors to show the scale of the Universe (i.e., plotted sphere)
    propsTop  = dict(mutation_scale=1, lw=lw, linestyle='solid', arrowstyle="-", color='w')
    propsBot  = dict(mutation_scale=1, lw=lw, linestyle='solid', arrowstyle="-", color='w')
    propsVert = dict(mutation_scale=mscl, lw=lw, linestyle='solid', arrowstyle="<|-|>", color='w')
    ax.annotate('', xy=(0.500,0.754), xytext=(0.717,0.800), xycoords='figure fraction', arrowprops=propsTop)
    ax.annotate('', xy=(0.611,0.312), xytext=(0.707,0.337), xycoords='figure fraction', arrowprops=propsBot)
    ax.annotate('', xy=(0.707,0.337), xytext=(0.717,0.800), xycoords='figure fraction', arrowprops=propsVert)

    # Print static text onto the figure
    textLtop = "The TIME is..."
    ax.annotate(textLtop, (0.15,0.675), xycoords='figure fraction', fontsize=fsS, color='w', ha='center', va='center', fontproperties=jsansSB)
    textRtop = "The SIZE is..."
    ax.annotate(textRtop, (0.85,0.675), xycoords='figure fraction', fontsize=fsS, color='w', ha='center', va='center', fontproperties=jsansSB)

    # Print the AstroSoM title and web address on the figure
    emDash   = u'\u2014'
    textASOM = "Astronomy Sound of the Month" + "\n" + emDash + " AstroSoM.com " + emDash
    ax.text(0.5, 0.075, textASOM, fontsize=fsXS, color='w', ha='center', va='center', transform=ax.transAxes, fontproperties=jsansR)
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # NOTE: Yes, I am repeating this twice, which is not good code practice or whatever.
    # But it gets the job done right now and I could clean it up by making a class later.

    # Set the starting azimuth and elevation
    azim0 = 0.0   # [degrees]
    elev0 = 60.0  # [degrees]

    # We will loop through each time/size in a smart way...
    # ...output entire set of frames where the CMB sphere has a given azimuthal angle
    N         = len(t_arr)
    Nz        = np.ceil(np.log10(N)).astype(int)
    N_arr     = np.arange(N)
    azim_arr  = np.empty(N)
    rout_list = []
    fout_list = []
    for i in N_arr:
        
        # Calculate the azimuthal angle for frame i [0,360]
        azim = azim0 + i * dtheta
        while (azim >= 360.0): azim = azim - 360.0
        azim_arr[i] = azim

        # Calculate the output file name for rendering i
        rout_list.append(rroot + "_" + str(i).zfill(Nz) + ".png")

        # Calculate the output file name for frame i
        fout_list.append(froot + "_" + str(i).zfill(Nz) + ".png")

    # Sort the indexing array by azimuthal angle of the CMB sphere rendering
    # NOTE: kind='mergesort' does exactly what I want! :)
    isort  = np.argsort(azim_arr, kind='mergesort')
    N_sort = N_arr[isort]
    
    #!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

    # Create each individual frame, taking advantage of multiple frames having the same azimuthal angle
    print("\nCreating individual frames...\n")
    icnt  = 1
    pctComplete = 0
    dummy = -1.0 * dtheta
    azim_old = dummy
    for i in N_sort:
        
        # Check if the previous frame has a different azimuthal angle from the current frame
        # (Hacky if statement to replace "if (azim_now != azim_old)", which is not reliable)
        azim_now = azim_arr[i]
        if (np.abs(azim_now - azim_old) > 0.5*dtheta):
        
            # Remove the old CMB sphere (unless one has not been rendered yet)
            if (azim_old != dummy): imcmb.remove()
            
            # Display the new CMB sphere on the 2D plot
            cmb = image.imread(rout_list[i])
            imcmb = ax.imshow(cmb)
        
        # Update azim_old
        azim_old = azim_now

        # Convert the current time and size to a string (rounding down)
        tnow = t_arr[i]
        anow = a_arr[i]

        # Print the time and size on the figure
        textLbot = timestr(tnow)
        textRbot = sizestr(anow)
        ttxt     = ax.annotate(textLbot, (0.15,0.5), xycoords='figure fraction', fontsize=fsS, color='w', ha='center', va='center', fontproperties=jsansR)
        atxt     = ax.annotate(textRbot, (0.85,0.5), xycoords='figure fraction', fontsize=fsS, color='w', ha='center', va='center', fontproperties=jsansR)

        # Save the figure
        fout = fout_list[i]
        fig.savefig(fout, facecolor=fig.get_facecolor(), edgecolor='none', dpi=dpi)

        # Remove the current time/size text output
        ttxt.remove()
        atxt.remove()
        
        # Give a progress report every so often
        if ((icnt % 100) == 0):
            pctComplete = float(icnt) / float(N) * 100
            print "    % Complete: ", '{:.1f}'.format(pctComplete)
        icnt = icnt + 1
    
    # Close the plot object
    plt.close()

