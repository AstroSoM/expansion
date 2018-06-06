import numpy as np

H0_Planck = 67.74  # km s^-1 Mpc^-1
H0_Riess  = 73.24  # km s^-1 Mpc^-1
H0        = (H0_Planck + H0_Riess) * 0.5
km2cm     = 1.0e5                          # [cm km^-1]
Mpc2pc    = 1.0e6                          # [pc Mpc^-1]
pc2cm     = 3.08567758149137e18            # [cm pc^-1]
H0cgs     = H0 * km2cm / (Mpc2pc * pc2cm)  # Hubble constant in CGS units [s^-1]
tHcgs     = 1.0 / H0cgs                    # Hubble time in CGS units [s]
sec2yr    = 1.0 / (3600 * 24 * 365.25)     # [yr s^-1]
yr2Gyr    = 1.0 / 1.0e9                    # [Gyr yr^-1]
tHGyr     = tHcgs * sec2yr * yr2Gyr        # Hubble time [Gyr]

def calc_t(lnN):
    t = lnN * tHGyr
    return t

def calc_N(t):
    N = np.exp(t / tHGyr)
    return N

# Size of the observable Universe today / size of a proton
ly2cm = 9.461e+17
fm2cm = 1e-13
rUniverse = 45.7e9 * ly2cm  # Radius of the observable Universe [cm]
rProton   = 0.8751 * fm2cm  # Radius of a proton [cm]
rU_rP     = rUniverse / rProton

# N = 2, 100, 10^100, (10^100)^100
ln2 = np.log(2)
ln100 = np.log(100)
lngoogol = np.log(1e100)
lngoogolplex = 2.30259e100

lnN = [ln2, ln100, np.log(rU_rP), lngoogol, lngoogolplex]
for i in np.arange(len(lnN)):
    t = calc_t(lnN[i])
    print "Scale Factor [-], Time [Gyr]", np.exp(lnN[i]), t

# How much does the Universe expand in 100 years?
tHyr   = tHGyr / yr2Gyr  # [yr]
a100yr = np.exp(100.0 / tHyr)
mi2in  = 63360.0       # [inches mile^-1]
dLA2NY = 2451 * mi2in  # [inches]
delta  = (a100yr - 1.0) * dLA2NY  # [inches]
print "\nIf you lived to be 100 years, the expansion of the Universe is analogous to the distance between LA and NYC expanding by [inches]: ", delta

print ""
