This directory contains the SDSS-V MWM training set data.
This data is to be used to help train a spectral "classifier"
that can figure out which of 10 coarse spectral class bins to
place individual spectra.   The information will then be used
to run more advanced analysis pipelines on those spectra.

There are currently 10 spectral classes, 6 optical and 4 NIR.
Here's a brief description of the datasets:

OPTICAL
=======

CV
-135 BOSS spectra provided by Boris Gaensicke (0.18GB)

WD single
-7,842 BOSS spectra provided by Boris Gaensicke (10.4GB)

WD binaries
-203 BOSS spectra provided by Boris Gaensicke (0.3GB)

HOTSTARS
-864 Combination of BOSS hot star spectra selected using stellar parameters provided by Young Sun Lee
      and MaStar hot star spectra selected using stellar parameters provided by David Nidever (1.1GB)

FGKM
-8,447 BOSS spectra selected using stellar parameters provided by Young Sun Lee (15.2GB)

YSO
-331 BOSS Orion spectra selected by Garrett Somers (0.3GB)


NIR
===

HOTSTARS
-93,240 synthetic spectra (23,310 for S/N=40,60,80,100) from Andrew Tkachenko (15.6GB)
-11,000 apVisit APOGEE spectra of ~2000 stars selected by David Nidever (5.5GB)

FGKM
-14,064 Combination of apVisit spectra of ~5000 APOGEE stars with good parameters and S/N>30 by David Nidever
          and apVisit spectra of very cool (3000-4000K) APOGEE stars selected by Mellisa Ness (6.9GB)

YSO
-6,836 apVisit spectra of ~3,000 YSOs provided by Marina Kounkel (3.4GB)

SB2
-4,227 apVisit spectra of ~1000 visually confirmed SB2s from Kevin Covey (2.0GB)


READING THE DATA
================
This directory contains a simple python module called loadspec.py.  You can
use this to load any of the training set spectra.  It will return a simple
Spec1D object that always has FILENAME, FLUX, ERR, MASK, WAVE and other values.
Here's how you use it:
% from loadspec import rdspec
% spec = rdspec(filename)


GETTING THE DATA
================
The easiest way to use the data is to obtain an account at Utah and use the files
directly on the Utah servers.  If that is not a possibility then there are a few ways
to download the files.
1) Tar files.  The spectra are also bundled into a number of compressed tar files (40GB total).
    The "tar.lst" list of tar files can be used to download all the files using
    "wget --user=USER --password=PASSWORD -i tar.lst" (you will need to use the provided USER/PASSWORD).
    You might need to use --no-check-certificate as well.
    The subdirectories are preserved in these tar files so you should just be able to untar them.
2) All individual files.  This directory contains a list of files for each training set
     (e.g., "nir_fgkm.lst").   These can be used with wget (as above) to download all the files.
     These will be put in the current working directory so it is advised that you first set up
     your own directory tree structure.
3) Individual files.  All of the individual files exist in this directory structure and you can
    search through and find the ones you want to download.