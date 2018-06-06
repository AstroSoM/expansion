import numpy as np
import healpy as hp
import h5py
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from mpl_toolkits.mplot3d import Axes3D, proj3d
from matplotlib import cm, colors
from matplotlib.patches import FancyArrowPatch

'''
Purpose:
--------
Plot the frames needed to make the video of the CMB on a sphere, accompanied by the evolution of time and the scale factor.

Notes:
------
I think Npix = 512 may be the limit of what we can feasibly render.
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

def bignumber(x):
    logx = np.log10(x)
    # Some big numbers
    hundred      = 2
    thousand     = 3
    million      = 6
    billion      = 9
    trillion     = 12
    quadrillion  = 15
    quintillion  = 18
    sextillion   = 21
    septillion   = 24
    octillion    = 27
    nonillion    = 30
    decillion    = 33
    undecillion  = 36
    duodecillion = 39
    tredecillion = 42
    quattuordecillion = 45
    quindecillion   = 48
    sexdecillion    = 51
    septendecillion = 54
    octodecillion   = 57
    novemdecillion  = 60
    vigintillion    = 63
    unvigintillion  = 66
    duovigintillion = 69
    trevigintillion = 72
    quattuorvigintillion = 75
    quinvigintillion   = 78
    sexvigintillion    = 81
    septenvigintillion = 84
    octovigintillion   = 87
    novemvigintillion  = 90
    trigintillion      = 93
    untrigintillion    = 96
    duotrigintillion   = 99
    googol             = 100
    if (logx < hundred): return '', 0
    if ((logx >= hundred) and (logx < thousand)): return 'hundred', hundred
    if ((logx >= thousand) and (logx < million)): return 'thousand', thousand
    if ((logx >= million) and (logx < billion)): return 'million', million
    if ((logx >= billion) and (logx < trillion)): return 'billion', billion
    if ((logx >= trillion) and (logx < quadrillion)): return 'trillion', trillion
    if ((logx >= quadrillion) and (logx < quintillion)): return 'quadrillion', quadrillion
    if ((logx >= quintillion) and (logx < sextillion)): return 'quintillion', quintillion
    if ((logx >= sextillion) and (logx < septillion)): return 'sextillion', sextillion
    if ((logx >= septillion) and (logx < octillion)): return 'septillion', septillion
    if ((logx >= octillion) and (logx < nonillion)): return 'octillion', octillion
    if ((logx >= nonillion) and (logx < decillion)): return 'nonillion', nonillion
    if ((logx >= decillion) and (logx < undecillion)): return 'decillion', decillion
    if ((logx >= undecillion) and (logx < duodecillion)): return 'undecillion', undecillion
    if ((logx >= duodecillion) and (logx < tredecillion)): return 'duodecillion', duodecillion
    if ((logx >= tredecillion) and (logx < quattuordecillion)): return 'tredecillion', tredecillion
    if ((logx >= quattuordecillion) and (logx < quindecillion)): return 'quattuordecillion', quattuordecillion
    if ((logx >= quindecillion) and (logx < sexdecillion)): return 'quindecillion', quindecillion
    if ((logx >= sexdecillion) and (logx < septendecillion)): return 'sexdecillion', sexdecillion
    if ((logx >= septendecillion) and (logx < octodecillion)): return 'septendecillion', septendecillion
    if ((logx >= octodecillion) and (logx < novemdecillion)): return 'octodecillion', octodecillion
    if ((logx >= novemdecillion) and (logx < vigintillion)): return 'novemdecillion', novemdecillion
    if ((logx >= vigintillion) and (logx < unvigintillion)): return 'vigintillion', vigintillion
    if ((logx >= unvigintillion) and (logx < duovigintillion)): return 'unvigintillion', unvigintillion
    if ((logx >= duovigintillion) and (logx < trevigintillion)): return 'duovigintillion', duovigintillion
    if ((logx >= trevigintillion) and (logx < quattuorvigintillion)): return 'trevigintillion', trevigintillion
    if ((logx >= quattuorvigintillion) and (logx < quinvigintillion)): return 'quattuorvigintillion', quattuorvigintillion
    if ((logx >= quinvigintillion) and (logx < sexvigintillion)): return 'quinvigintillion', quinvigintillion
    if ((logx >= sexvigintillion) and (logx < septenvigintillion)): return 'sexvigintillion', sexvigintillion
    if ((logx >= septenvigintillion) and (logx < octovigintillion)): return 'septenvigintillion', septenvigintillion
    if ((logx >= octovigintillion) and (logx < novemvigintillion)): return 'octovigintillion', octovigintillion
    if ((logx >= novemvigintillion) and (logx < trigintillion)): return 'novemvigintillion', novemvigintillion
    if ((logx >= trigintillion) and (logx < untrigintillion)): return 'trigintillion', trigintillion
    if ((logx >= untrigintillion) and (logx < duotrigintillion)): return 'untrigintillion', untrigintillion
    if ((logx >= duotrigintillion) and (logx < googol)): return 'duotrigintillion', duotrigintillion
    if (logx >= googol): return 'googol', googol

# Do time in billions of years
def timestr(t):
    tdiv = t / 1e9
    tstr = str(np.floor(tdiv).astype(int))
    ttxt = r"$\mathregular{" + tstr + "}$" + "\n" + "billion" + "\n" + "years from today"
    return ttxt

def sizestr(a):
    name, power = bignumber(a)
    adiv = a / 10.0**power
    astr = str(np.floor(adiv).astype(int))
    atxt = r"$\mathregular{" + astr + "}$" + "\n" + name + "\n" + "times that today"
    return atxt

#====================================================================================================
def cmb(t_arr, a_arr, Npix=32, Nrot=1, pixres=1080, ffits="data/COM_CMB_IQU-commander-field-Int_2048_R2.01_full.fits", froot="/Users/salvesen/outreach/asom/expansion/results/frames/cmb"):
    '''
    Inputs:
    -------
    t_arr  - Array of times [Gyr]
    a_arr  - Array of scale factors/sizes [-]
    Npix   - Number of (phi, theta) pixels to use (2048 max)
    Nrot   - Number of full rotations to make of the CMB sphere
    pixres - Image resolution (240p, 360p, 480p, 720p, 1080p, etc.)
    ffits  - CMB map FITS file (https://pla.esac.esa.int/pla/)
    froot  - Root name for the output PNG files
    '''
    
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
    colombi1_cmap = colors.ListedColormap(np.loadtxt("Planck_Parchment_RGB.txt") / 255.0)
    colombi1_cmap.set_bad("gray")     # Color of missing pixels
    colombi1_cmap.set_under("white")  # Color of background, necessary if you want to use this colormap directly with hp.mollview(m, cmap=colombi1_cmap)
    cmap = colombi1_cmap

    # Create a sphere
    r = 1.0
    x = r * np.sin(THETA) * np.cos(PHI)
    y = r * np.sin(THETA) * np.sin(PHI)
    z = r * np.cos(THETA)
    rmin = -1.0 * r
    rmax = r

    # Plotting preferences
    fres  = pixres / 720.0
    dpi   = 200
    lw    = 1.333 * fres
    fsL   = 32 * fres
    fsS   = 16 * fres
    fsXS  = 10 * fres
    mscl  = 16 * fres
    left, right, bottom, top = 0.0, 1.0, 0.0, 1.0
    xsize, ysize = XYsize(pixres=pixres, dpi=dpi)
    plt.rcParams['axes.linewidth'] = lw
    plt.rcParams['axes.facecolor'] = 'k'
    plt.rcParams['axes.titlepad']  = -1.0*fsS

    # Font
    jsansR  = fm.FontProperties(fname='fonts/JosefinSans-Regular.ttf')
    jsansB  = fm.FontProperties(fname='fonts/JosefinSans-Bold.ttf')
    jsansSB = fm.FontProperties(fname='fonts/JosefinSans-SemiBold.ttf')
    jsansBI = fm.FontProperties(fname='fonts/JosefinSans-BoldItalic.ttf')

    # Initialize the figure and do various aesthetic things
    fig = plt.figure(figsize=(xsize, ysize), dpi=dpi)
    fig.subplots_adjust(left=left, right=right, bottom=bottom, top=top)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlim(rmin, rmax)
    ax.set_ylim(rmin, rmax)
    ax.set_zlim(rmin, rmax)
    ax.set_title("The Universe", fontsize=fsL, fontproperties=jsansB, color='w')

    # Get rid of the panes
    ax.w_xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the spines
    ax.w_xaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_yaxis.line.set_color((1.0, 1.0, 1.0, 0.0))
    ax.w_zaxis.line.set_color((1.0, 1.0, 1.0, 0.0))

    # Get rid of the ticks
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_zticks([])

    # Draw lines/vectors to show the scale of the Universe (i.e., plotted sphere)
    propsTop  = dict(mutation_scale=1, lw=lw, linestyle='solid', arrowstyle="-", color='w')
    propsBot  = dict(mutation_scale=1, lw=lw, linestyle='solid', arrowstyle="-", color='w')
    propsVert = dict(mutation_scale=mscl, lw=lw, linestyle='solid', arrowstyle="<|-|>", color='w')
    ax.annotate('', xy=(0.506,0.754), xytext=(0.717,0.800), xycoords='figure fraction', arrowprops=propsTop)
    ax.annotate('', xy=(0.611,0.312), xytext=(0.707,0.337), xycoords='figure fraction', arrowprops=propsBot)
    ax.annotate('', xy=(0.707,0.337), xytext=(0.717,0.800), xycoords='figure fraction', arrowprops=propsVert)
    '''
    # The method below rotates the arrow with the sphere, which is not what we want for this case.
    class Arrow3D(FancyArrowPatch):
        def __init__(self, xs, ys, zs, *args, **kwargs):
            FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
            self._verts3d = xs, ys, zs
        def draw(self, renderer):
            xs3d, ys3d, zs3d = self._verts3d
            xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
            self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
            FancyArrowPatch.draw(self, renderer)
    atop  = Arrow3D([0, 1], [0, 1], [1, 1], mutation_scale=1, lw=lw, arrowstyle="-", color='r')
    abot  = Arrow3D([0.5, 1], [0.5, 1], [-1, -1], mutation_scale=1, lw=lw, arrowstyle="-", linestyle='solid', color='r')
    avert = Arrow3D([1, 1], [1, 1], [-1, 1], mutation_scale=mscl, lw=lw, arrowstyle="<|-|>", color='r')
    ax.add_artist(atop)
    ax.add_artist(abot)
    ax.add_artist(avert)
    '''
    
    # Draw the CMB sphere
    print("\nRendering the CMB sphere...\n")
    #surf = ax.plot_surface(x, y, z, facecolors=cm.jet(norm(grid_map)), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=True)
    surf = ax.plot_surface(x, y, z, facecolors=cmap(norm(grid_map)), rstride=1, cstride=1, linewidth=0, antialiased=False, shade=True)

    # Print static text onto the figure
    textLtop = "The TIME is..."
    ax.annotate(textLtop, (0.15,0.65), xycoords='figure fraction', fontsize=fsS, color='w', ha='center', va='center', fontproperties=jsansSB)
    textRtop = "The SIZE is..."
    ax.annotate(textRtop, (0.85,0.65), xycoords='figure fraction', fontsize=fsS, color='w', ha='center', va='center', fontproperties=jsansSB)

    # Print the AstroSoM title and web address on the figure
    emDash   = u'\u2014'
    textASOM = "Astronomy Sound of the Month" + "\n" + emDash + " AstroSoM.com " + emDash
    ax.text(0.0, 0.0, -1.75,  textASOM, fontsize=fsXS, color='w', ha='center', va='center', fontproperties=jsansR)

    #----------------------------------------------------------------------------------------------------

    # Initialize the viewing angle
    azim0 = -60.0  # Starting azimuth [degrees]
    elev0 = 30.0   # Starting elevation [degrees]
    ax.view_init(azim=azim0, elev=elev0)

    # Loop through each time and size
    N  = len(t_arr)
    Nz = np.ceil(np.log10(N)).astype(int)
    dtheta   = (360.0 / (N - 1.0)) * Nrot  # [degrees]
    azim_old = azim0
    azim_now = azim0
    print("\nCreating individual frames...\n")
    for i in np.arange(N):

        # Convert the current time and size to a string (rounding down)
        tnow = t_arr[i]
        anow = a_arr[i]

        # Print the time and size on the figure
        textLbot = timestr(tnow)
        textRbot = sizestr(anow)
        ttxt     = ax.annotate(textLbot, (0.15,0.5), xycoords='figure fraction', fontsize=fsS, color='w', ha='center', va='center', fontproperties=jsansR)
        atxt     = ax.annotate(textRbot, (0.85,0.5), xycoords='figure fraction', fontsize=fsS, color='w', ha='center', va='center', fontproperties=jsansR)

        # Viewpoint for the CMB sphere
        azim = azim0 + i * dtheta
        ax.view_init(azim=azim, elev=elev0)
        '''
        # The method below of rendering rotations periodically does not reduce computation time :(
        if ((azim_now - azim_old) >= Ndeg):
            ax.view_init(azim=azim_now, elev=elev0)
            azim_old = azim_now
        azim_now += dtheta
        '''
        
        # Save the figure
        fout = froot + "_" + str(i).zfill(Nz) + ".png"
        fig.savefig(fout, facecolor=fig.get_facecolor(), edgecolor='none', dpi=dpi)

        # Remove only the current time/size text output
        ttxt.remove()
        atxt.remove()

        # Give a progress report every so often
        if ((i % 10) == 0):
            pctComplete = float(i+1) / float(N) * 100
            print "    % Complete: ", '{:.1f}'.format(pctComplete)

    #----------------------------------------------------------------------------------------------------

    # Close the plot object
    plt.close()

#====================================================================================================

# GENERATE THE FRAMES

H0     = 70.49                          # Hubble constant [km s^-1 Mpc^-1]
km2cm  = 1.0e5                          # [cm km^-1]
Mpc2pc = 1.0e6                          # [pc Mpc^-1]
pc2cm  = 3.08567758149137e18            # [cm pc^-1]
H0cgs  = H0 * km2cm / (Mpc2pc * pc2cm)  # Hubble constant in CGS units [s^-1]
tHcgs  = 1.0 / H0cgs                    # Hubble time in CGS units [s]
sec2yr = 1.0 / (3600 * 24 * 365.25)     # [yr s^-1]
yr2Gyr = 1.0 / 1.0e9                    # [Gyr yr^-1]
tHGyr  = tHcgs * sec2yr * yr2Gyr        # Hubble time [Gyr]
tH     = tHGyr * 1.0e9                  # Hubble time [yr]

fps  = 6     # Frame rate [frames/second]
mpr  = 6     # Minutes per rotation <-- 1 full rotation every mpr minutes
tmin = 0     # Start time [Gyr]
tmax = 3194  # End time [Gyr]
Nt   = int(fps * (tmax - tmin) + 1)  # Number of frames
Nsec = float(Nt-1) / float(fps)      # Length of the output movie [seconds]
Nrot = Nsec / (mpr * 60.0)           # Number of rotations (total)

# Time and Size arrays
t_arr = np.linspace(tmin, tmax, Nt) * 1.0e9  # [yr]
a_arr = np.exp(t_arr / tH)

# Generate the plots
cmb(t_arr=t_arr, a_arr=a_arr, Npix=1024, Nrot=Nrot, pixres=1080)

# Stitch the frames together into a video
#ffmpeg -f image2 -framerate 6 -i 'cmb_%04d.png' -s 1920X1080 -pix_fmt yuv420p cmb.mp4

# Combine the video and audio
#ffmpeg -i cmb.mp4 -i risset.wav -c:v copy -c:a aac -strict experimental expansion.mp4

