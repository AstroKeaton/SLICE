'''
Code originally written in MATLAB by: A. Bolatto

Adapted and modified for use in Python by: K. Donaghue
Last modified: 2/20/2026

This code is best applied to JWST or similar IR data

Primary functions:
-catPeaks
    -Take fits cubes and extract spectra of set region
    -Identify peaks in spectrum and fit gaussian to them
    -Record key properties of lines such as intensity and flux
    -Produces among other things a line list to be used for IDs
    -1D spectra may also be inputted to find peaks
    -An annulus around the aperture may be defined for a sky subtraction
-keepLines
    -Find and keep lines near specific wavelengths given some tolerance
-cullLines
    -Remove rows(i.e. specific lines) from line list
-lineID
    -Parses through line list and identifies lines 
    -Uses multiple line lists to ID lines 
    -Yields likely lines for peak in order of best fit
-findLines
    -Functions similar to catPeaks but for a specified wavelength range.
    -Optionally performs lineID on new wavelength range for identifying lines.
-findSpecies
    -Finds and reports all lines of a given type and optionally transition in speclist.
    -May optionally plot the identified lines against the spectra.
-readLineID
    -Helper to read the lineID .txt file.
    -Creates a dictionary of key parameters to call from the ID list (i.e. rest or obs wavelength).
-plotID
    -quick plotting tool for plotting the spectrum with identified lines marked and labeled
-combList
    -combines lists into a master list i.e. all 4 MIRI MRS Channel data into one list
-getLines
    -Retrive specific lines or species from list i.e. all HI lines
'''


import numpy as np
import re
from typing import List, Dict, Any
import os
import IPython
import logging
from logging import config



########################################################
########Set up logging##################################
def _logging_setup():
    ''' Set up logs with python's logging module. 
    Checks for user-defined logger configuration file named "logger.conf" in the current working directory.
    Otherwise, sends ERROR+ to terminal and INFO+ to log_file. Log files will be saved in the cwd/SLICE_logs/ and are named as "SLICE_YYYY-MM-DD.log" in UTC. Executions run on the same date will be appended to the end of the (existing) file.
    
    Returns
    -------
    logger : object
        root level logger object
    '''
    import os
    import datetime
    from datetime import timezone

    #check if a user-defined logeer config file exists
    if os.path.isfile("logger.conf"):
        config.fileConfig("logger.conf")
        user_config = True

    else: #use default
        user_config = False
        log_file_path = "SLICE_logs/"
        if os.path.isdir(log_file_path) == False:
            os.makedirs(log_file_path, exist_ok=True)
        today = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d")
        log_file = log_file_path+"SLICE_"+today+".log"
        if os.path.isfile(log_file) == True:
            fmode = 'a'
        else:
            fmode = 'w'

        log_config = {
            "version":1,
            "root":{
                "handlers" : ["console", "file"],
                "level": "DEBUG"
            },
            "handlers":{
                "console":{
                    "formatter": "basicFormatter",
                    "class": "logging.StreamHandler",
                    "level": "ERROR"
                },
                "file":{
                    "formatter":"basicFormatter",
                    "class":"logging.FileHandler",
                    "level":"INFO",
                    "filename":log_file,
                    "mode":fmode
                }
            },
            "formatters":{
                "basicFormatter": {
                    "format": "%(levelname)s : %(module)s : %(funcName)s : %(message)s",
                }
            },
        }       

        #save default config to a file
        with open(log_file_path+"default_logger.conf","w") as fp:
            print(log_config,file=fp)

        config.dictConfig(log_config)

    #pass items from python warnings to logging instead
    logging.captureWarnings(True)

    logger = logging.getLogger(__name__)

    #print date and time of script execution to log file
    now = datetime.datetime.now(timezone.utc).strftime("%Y-%m-%d_%H:%M:%S")
    logger.info("\n\n%s\nExecuted from script in %s/ on %s at %s UTC\n%s\n\n"
        %("#"*42,os.path.abspath(os.getcwd()),now.split("_")[0],now.split("_")[1],"#"*42))

    if user_config == True:
        logger.info("Using logger configuration from %s/logger.conf." %os.path.abspath(os.getcwd()))
    else:
        logger.info("Using default logger configuration, which has been saved in %s." %(log_file_path+"default_logger.conf"))

    return logger
logger = _logging_setup()

########Helper functions for catPeaks##################################
def _normalize_cube_to_yxspec(data, wcs1, wcs2, spec_len):
    """
    Reorder a data cube so it is (ny, nx, nspec), matching wcs1/wcs2 shapes.
    This is how later code assumes the shape
    """
    import numpy as np
    ny, nx = wcs1.shape
    shape = data.shape

    # pick spectral axis by length match; fallback to the largest axis
    cand = [i for i, s in enumerate(shape) if s == spec_len]
    spec_axis = cand[0] if cand else int(np.argmax(shape))

    # move spectral axis to last
    cube = np.moveaxis(data, spec_axis, -1)  # -> (*spatial*, nspec)

    # ensure spatial order matches WCS (ny, nx)
    if cube.shape[:2] == (ny, nx):
        return cube
    elif cube.shape[:2] == (nx, ny):
        return cube.transpose(1, 0, 2)       # swap x/y
    else:
        raise ValueError(
            f"Spatial dims {cube.shape[:2]} don't match WCS {wcs1.shape}."
        )



def gaussmfit(x, y, K, S, X0, bounds=None, tol=1e-7, max_iter=200, lambda_init=1e-3, verbose=False, showfit = False):
    """
    Fit one or more Gaussian functions to data using Levenberg–Marquardt optimization.
    
    Parameters
    ----------
    x, y : array_like
        Data points (1D arrays of equal length).
    K : array_like
        Initial amplitudes of Gaussians.
    S : array_like
        Initial standard deviations of Gaussians.
    X0 : array_like
        Initial centers of Gaussians.
    bounds : array_like
        Array containing lists the lower and upper bounds for each parameter (e.g., [[K_min, S_min, X0_min],[K_max, S_max, X0_max]]) (Default: None)
    tol : float, optional
        Convergence tolerance on chi-square reduction.
    max_iter : int, optional
        Maximum number of iterations.
    lambda_init : float, optional
        Initial damping parameter for LM algorithm.
    verbose : bool, optional
        Print iteration details if True.
    
    Returns
    -------
    K, S, X0 : ndarray
        Best-fit Gaussian parameters.
    chi2 : float
        Reduced chi-square of the fit.
    ymodel : ndarray
        Best-fit model evaluated at x.
    err_code : int
        Error flag (0 = success, 1 = L exceeded, 5 = singular matrix).
    epar : ndarray
        1-sigma uncertainties of fitted parameters.
    """
    
    import numpy as np
    import matplotlib.pyplot as plt
    from numpy.linalg import solve, inv
    
    # Convert inputs
    x, y = np.asarray(x, float), np.asarray(y, float)
    K_og, S_og, X0_og = np.asarray(K, float), np.asarray(S, float), np.asarray(X0, float)
    K, S, X0 = np.asarray(K, float), np.asarray(S, float), np.asarray(X0, float)
    b, d = len(K), len(x)
    
    if bounds == None:
        bounds = [[0., 0., -np.inf],[np.inf, np.inf, np.inf]]

    bounds = np.array(bounds)
    bK = bounds[:,0]
    bS = bounds[:,1]
    bX0 = bounds[:,2]

    # Expand to matrices for vectorized math
    x_mat = np.tile(x, (b, 1))
    y_mat = np.tile(y, (b, 1)) / b

    def model(Kv, Sv, X0v):
        """Gaussian model (vectorized for multiple peaks)."""
        return Kv[:, None] * np.exp(-((x_mat - X0v[:, None])**2) / (2 * Sv[:, None]**2))

    # Initial model and chi2
    ymodel = model(K, S, X0)
    ymodel_og = model(K_og, S_og, X0_og)
    delta = y_mat - ymodel
    chi2 = np.sum(delta**2)
    chi2_og = chi2.copy()
    # Damping parameter
    L = lambda_init
    err_code = 0
   
    # Main LM loop
    for it in range(max_iter):
        # Derivatives
        derK = ymodel / K[:, None]
        derS = ymodel * ((x_mat - X0[:, None])**2) / (S[:, None]**3)
        derX0 = ymodel * (x_mat - X0[:, None]) / (S[:, None]**2)

        # Build alpha and beta matrices
        alpha = np.zeros((3*b, 3*b))
        beta = np.zeros((3*b, 1))
        for i in range(b):
            dKi, dSi, dX0i = derK[i], derS[i], derX0[i]
            beta[3*i:3*i+3, 0] = [np.sum(delta[i] * dKi),
                                  np.sum(delta[i] * dSi),
                                  np.sum(delta[i] * dX0i)]
            for j in range(b):
                dKj, dSj, dX0j = derK[j], derS[j], derX0[j]
                diagV = (i == j)
                alpha[3*i:3*i+3, 3*j:3*j+3] = np.array([
                    [np.sum(dKi * dKj *(1+L*diagV)), np.sum(dKi * dSj), np.sum(dKi * dX0j)],
                    [np.sum(dSi * dKj), np.sum(dSi * dSj * (1+L*diagV)), np.sum(dSi * dX0j)],
                    [np.sum(dX0i * dKj), np.sum(dX0i * dSj), np.sum(dX0i * dX0j * (1+L*diagV))]
                ])

        # Solve for parameter step
        if np.linalg.cond(alpha) > 1e20:
            err_code = 5  # Singular matrix
            break
        da = solve(alpha, beta).ravel()

        # Update parameters
        newK = K + da[0::3]
        newS = S + da[1::3]
        newX0 = X0 + da[2::3]

        # Evaluate new model
        newymodel = model(newK, newS, newX0)
        newdelta = y_mat - newymodel
        newchi2 = np.sum(newdelta**2)
        
         #get conditionals to accept step
        accept_cond = (newchi2 < chi2) and np.all(newK >= bK[0]) and np.all(newK <= bK[1]) and np.all(newS >= bS[0]) and np.all(newS <= bS[1]) and np.all(newX0 >= bX0[0]) and np.all(newX0 <= bX0[1])

        if accept_cond:  # Accept step
            converge_val = (chi2 - newchi2) / newchi2
            if verbose:
                print(f"Iter {it:3d} | chi2={newchi2:.4e} | Δχ²/χ²={converge_val:.2e} | L={L:.2e}")
            L /= 10
            K, S, X0, ymodel, delta, chi2 = newK, newS, newX0, newymodel, newdelta, newchi2
            if converge_val < tol:
                break  # Converged
        else:  # Reject step
            L *= 10
            if L > 1e3:
                err_code = 1  # L exceeded
                break

    # Collapse multi-component fit into final ymodel
    if b > 1:
        ymodel_sum = np.sum(ymodel, axis=0)
        ymodel_og_sum = np.sum(ymodel_og, axis=0)
    else:
        ymodel_sum = ymodel
        ymodel_og_sum = ymodel_og

    # Reduced chi-square
    dof = max(1, d - 3*b)
    chi2 /= dof

    # Compute covariance matrix and parameter uncertainties
    emat = inv(alpha)
    epar = np.sqrt(chi2 * np.abs(np.diag(emat)))
    
    
    if showfit is True:
        plt.plot(x,y, label='spectra')
        if np.shape(ymodel) == (1, 31):
            plt.plot(x, ymodel_sum[0], label = 'model')
        else:
            plt.plot(x, ymodel_sum, label='model')
        if np.shape(ymodel_og_sum) == (1, 31):
            plt.plot(x,ymodel_og_sum[0], label = 'Guess fit')
        else:
            plt.plot(x, ymodel_og_sum, label = 'Guess fit')
        
        if np.shape(ymodel) == (3, 31):
            plt.plot(x,ymodel[0], label='first component')
            plt.plot(x,ymodel[1], label='second component')
            plt.plot(x,ymodel[2], label='third component')
        plt.legend()
        plt.show()

    #if err_code == 1 or err_code == 5:  
        #IPython.core.debugger.set_trace()
    return K, S, X0, chi2, ymodel, err_code, epar



def sex2dec(sexstr):
    """
    Convert sexagesimal string to decimal degrees.

    Parameters:
    ------------
    sexstr : str
        assumes form of 'HH:MM:SS.S' or '±DD:MM:SS.S'

    Returns:
    ---------
    float : value in decimal degrees
    """
    if isinstance(sexstr, str):
        parts = sexstr.strip().split(':')
        if len(parts) != 3:
            raise ValueError(f"Invalid sexagesimal string: {sexstr}")

        sign = -1 if sexstr.strip().startswith('-') else 1
        h_or_d = abs(float(parts[0]))
        m = float(parts[1])
        s = float(parts[2])

        return sign * (h_or_d + m / 60.0 + s / 3600.0)
    else:
        raise TypeError("Input must be a string in 'HH:MM:SS' format")

def rfits(file, option=None, bval=np.nan):
    """
    Fits reader for acquiring necessary data for functions.
    
    Parameters
    -----------
    file : (str) 
        Path to FITS file.
    option: (str or None) 
        Optional keywords (e.g. 'head', 'nowcs', 'noabs', 'xten1', etc.).
    bval : (float)
        Value to use for blank pixels. Default is nan.
    
    Returns:
    --------
    dict: containing data, header, axes, and WCS info.
    """
    from astropy.io import fits
    from astropy.wcs import WCS
    import numpy as np
    
    header_only = 'head' in option if isinstance(option, str) else False
    nowcs = 'nowcs' in option if isinstance(option, str) else False
    noabs = 'noabs' in option if isinstance(option, str) else False
    verbose = 'verbose' in option if isinstance(option, str) else False
    extension = 0

    # Handle extension (e.g. 'xten1' → extension 1)
    if isinstance(option, str) and 'xten' in option:
        try:
            extension = int(option[option.find('xten') + 4])
        except Exception:
            extension = 0

    with fits.open(file, memmap=True) as hdul:
        hdu = hdul[extension]
        header = hdu.header

        if header_only:
            return {"header": header}

        data = hdu.data.astype(float)  # ensure float for blank handling
        if "BLANK" in header:
            blankval = header["BLANK"]
            data[data == blankval] = bval

        # Scale if needed
        bscale = header.get("BSCALE", 1.0)
        bzero = header.get("BZERO", 0.0)
        data = bscale * data + bzero

        # Extract WCS
        if not nowcs:
            try:
                wcs = WCS(header)
            except Exception:
                wcs = None
        else:
            wcs = None

        # Create axes
        naxis = header.get("NAXIS", 0)
        axes = []
        for i in range(naxis):
            axis_len = header.get(f"NAXIS{i+1}", 1)
            crval = header.get(f"CRVAL{i+1}", 0.0)
            cdelt = header.get(f"CDELT{i+1}", 1.0)
            crpix = header.get(f"CRPIX{i+1}", 1.0)
            axis = (np.arange(axis_len) + 1 - crpix) * cdelt
            if not noabs:
                axis += crval
            axes.append(axis)

        result = {
            "header": header,
            "data": data,
            "x": axes,
            "bunit": header.get("BUNIT", ""),
            "naxis": naxis,
            "bitpix": header.get("BITPIX", None),
            "numpt": [header.get(f"NAXIS{i+1}", 0) for i in range(naxis)],
            "crval": [header.get(f"CRVAL{i+1}", 0.0) for i in range(naxis)],
            "crpix": [header.get(f"CRPIX{i+1}", 1.0) for i in range(naxis)],
            "cdelt": [header.get(f"CDELT{i+1}", 1.0) for i in range(naxis)],
            "cunit": [header.get(f"CUNIT{i+1}", "") for i in range(naxis)],
            "ctype": [header.get(f"CTYPE{i+1}", "") for i in range(naxis)],
            "wcs": wcs,
            "coormode": "ABSCOOR" if not noabs else "RELCOOR",
        }

        # Optional WCS coordinate arrays
        if wcs is not None and naxis >= 2:
            ny, nx = data.shape[-2], data.shape[-1]
            yy, xx = np.mgrid[:ny, :nx]
            try:
                ra, dec = wcs.celestial.pixel_to_world(xx, yy).ra.deg, \
                          wcs.celestial.pixel_to_world(xx, yy).dec.deg
                result["wcs1"] = ra
                result["wcs2"] = dec
            except Exception as e:
                logger.error(f"WCS conversion failed: {e}")

        return result
    
    

    
def psfranger(wave,scale, mode=None):
    
    '''
    designed to scale the radius of the aperture for SLICE to account for the PSF
    
    Parameters
    --------------
    wave : float
        The wavelength of the spectrum in microns
    scale : float
        The desired scale factor to scale the fwhm by
     mode : str
        The desired mode for accounting for the PSF
   
   Current available modes:
        MRSFWHMLaw2023- MIRI MRS FWHM from Law et al (2023) good for point sources
        
    Returns
    --------------
    fin : float
        The radius of the aperture in arcsecounds to account for the PSF
    '''
    if mode is None:
        raise ValueError('Must specify what mode for the PSF')
    
    if mode is 'MRSFWHMLaw2023':
        size = 0.033 * (wave) + 0.106
        fin = size * scale
    else:
        raise ValueError('Unsupported mode specified')
    
    return(fin)


def aperture(x, y, xc, yc, psize, rad):
    """
    Compute the fractional contribution of each square pixel (of size psize)
    in an image to a circular aperture centered on (xc, yc) of radius rad.
    
    function is used in extract_aperture
    
    Parameters:
    x, y   : arrays of pixel center coordinates (same shape)
    xc, yc : coordinates of the circular aperture center
    psize  : pixel size (assumes square pixels)
    rad    : aperture radius

    Returns:
    weight : fractional weight array (same shape as x and y)
    """
    import numpy as np
    
    weight = np.zeros_like(x, dtype=float)

    # Distance from pixel corners to the aperture center
    c1 = np.sqrt((x + psize/2 - xc)**2 + (y + psize/2 - yc)**2)
    c2 = np.sqrt((x - psize/2 - xc)**2 + (y + psize/2 - yc)**2)
    c3 = np.sqrt((x - psize/2 - xc)**2 + (y - psize/2 - yc)**2)
    c4 = np.sqrt((x + psize/2 - xc)**2 + (y - psize/2 - yc)**2)
    
    rad = float(rad)
    nc = (c1 <= rad).astype(int) + (c2 <= rad).astype(int) + \
         (c3 <= rad).astype(int) + (c4 <= rad).astype(int)

    weight[nc == 4] = 1
    weight[nc == 0] = 0

    # Find partially overlapping pixels
    partial_mask = (nc > 0) & (nc < 4)
    ix = np.where(partial_mask.flatten())[0]
    nsamp = 10000
    xyr = psize * (np.random.rand(nsamp, 2) - 0.5)

    x_flat = x.flatten()
    y_flat = y.flatten()
    weight_flat = weight.flatten()

    for i in ix:
        dx = x_flat[i] + xyr[:, 0] - xc
        dy = y_flat[i] + xyr[:, 1] - yc
        inside = (dx**2 + dy**2) <= rad**2
        weight_flat[i] = np.sum(inside) / nsamp

    return weight.reshape(x.shape)


def annulus_weight(x, y, xc, yc, psize, rin, rout):
    """Fractional pixel weights for an annulus (rin < r < rout)."""
    w_out = aperture(x, y, xc, yc, psize, rout)
    w_in  = aperture(x, y, xc, yc, psize, rin)
    w = w_out - w_in
    w[w < 0] = 0
    return w

def sector_mask(xx, yy, center_x, center_y, exclude_sectors):
    """
    exclude_sectors = list of (theta_min, theta_max) in degrees
    angles measured CCW from +x axis
    """
    # Compute angle map in degrees
    theta = np.degrees(np.arctan2( (yy - center_y), -(xx - center_x) )) % 360

    mask = np.ones_like(theta)

    for tmin, tmax in exclude_sectors:
        if tmin < tmax:
            bad = (theta >= tmin) & (theta <= tmax)
        else:
            # wrap-around case (e.g., 350°–20°)
            bad = (theta >= tmin) | (theta <= tmax)

        mask[bad] = 0

    return mask


def extract_aperture(r, pixco, scale = 1.5, sky_annulus=None, exclude_sectors=None, useapprox=False, silent=False):
    """
    Extract a spectrum using an aperture on the FITS data cube.
    
    function is used for catPeaks
    
    Parameters:
    r : dict
        dictionary containing FITS data and WCS (from rfits)
    pixco : list or array 
        pixel coordinates, list of coordinates, or circular region spec
    scale : float 
        factor to scale the FWHM by if using the auto conical apperture scheme
    sky_annulus : float
        If None no sky annulus will be subtracted from the spectrum. when given a number (i.e. 2) 
        will creaate an annulus with inner radius of the number times the radius used for science aperture
        will extend to 1 + the number given (ex. input of 2 will have outer radius 3x the aperture radius.
    useapprox : bool
        if True, uses crude radial cutoff instead of accurate fractional aperture
    silent : bool
        if True, suppresses warnings

    Returns:
    s : array
        extracted 1D spectrum
    region : str
        string identifier for aperture type
    mask : array
        2D mask of used pixels
    flag : float
        status flag (0 = OK, -1 = outside cube, -2 = too many NaNs)
    """
    import numpy as np
    from numpy import cos, deg2rad
    from astropy.wcs.utils import skycoord_to_pixel
    from astropy.coordinates import SkyCoord
    from astropy.stats import sigma_clipped_stats
    import matplotlib.pyplot as plt
    import warnings
    
    #make sure pixco is properly defined
    if pixco is not None:
        if type(pixco) == list and len(pixco) == 1:
            raise ValueError('pixco list must specify x and y pixel coordinates')
        elif type(pixco) == tuple and np.shape(pixco)[0] > 2:
            raise ValueError('array list of pixel coordinates must be given as ([x1, x2, x3...],[y1, y2, y3...]')
    
    flag = 0
    data = r["data"]
    shape = data.shape
    
    # Reorder cube to (ny, nx, nspec) so data[:, :, k] is a spatial plane
    if ("wcs1" in r) and ("wcs2" in r) and ("x" in r) and (len(r["x"]) >= 3):
        data = _normalize_cube_to_yxspec(data, r["wcs1"], r["wcs2"], len(r["x"][2]))
        shape = data.shape
    else:
        # fallback: assume last axis is spectral
        pass

    
    mask = np.zeros(shape[:2], dtype=float)
    
    # Handle 'all' case (scalar 0)
    if isinstance(pixco, (int, float)) and pixco == 0:
        region = "all"
        s = np.nanmedian(data, axis=(0, 1))
        mask[:] = 1
        
    else:
        pixco = np.array(pixco)
        # Handle 'pixel' case
        if pixco.ndim == 1 and pixco.shape[0] == 2:
            region = "pixel"
            s = data[pixco[0], pixco[1], :]
            mask[pixco[0], pixco[1]] = 1

        # Handle 'all' case
        elif type(pixco) is int and pixco == 0:
            region = "all"
            s = np.nanmedian(data, axis=(0, 1))
            mask[:] = 1

        # Handle 'list' case
        elif pixco.shape[0] == 2 and pixco.shape[1] > 1:
            region = 'list'
            s = np.nanmedian(data[pixco[0], pixco[1], :], axis=(0,))
            mask[pixco[0], pixco[1]] = 1



        # Handle 'circle' case
        elif pixco.shape == (3,) or len(pixco) == 3:
            region = 'circle'
            if isinstance(pixco[0], str):
                rac = sex2dec(pixco[0]) * 15
                dec = sex2dec(pixco[1])
                rad = pixco[2]
            elif isinstance(pixco[0], (int, float)):
                rac, dec, rad = pixco
            else:
                raise ValueError("Format should be ['00:00:00.0','00:00:00.0', 0.0] or [0.0, 0.0, 0.0] for circular aperture")

            wcs1 = r["wcs1"]
            wcs2 = r["wcs2"]

            xx = (wcs1 - rac) * np.cos(np.deg2rad(r["crval"][1])) * 3600
            yy = (wcs2 - dec) * 3600

            s = np.zeros(shape[2])
            ck = np.zeros(shape[2])
            omega = np.zeros(shape[2])

            for k in range(shape[2]):
                wave = r["x"][2][k]  # wavelength in microns
                
                # --- choose aperture radius ---
                if rad == "auto":
                    rad_arcsec = psfranger(wave,scale, mode="MRSFWHMLaw2023")
                else:
                    rad_arcsec = float(rad)   # user-specified arcsec
                
                if useapprox:
                    dist = np.sqrt(xx**2 + yy**2)
                    ix = np.where(dist <= rad_arcsec)
                    p = data[:, :, k]
                    s[k] = np.nanmean(p[ix])
                else:
                    weight = aperture(xx, yy, 0, 0, r["cdelt"][1] * 3600, rad_arcsec)
                    omega[k] = np.nansum(weight) * (r["cdelt"][1] * np.pi / 180)**2

                    sw = weight / np.nansum(weight)

                    p = data[:, :, k] * sw
                    pck = np.ones_like(p)
                    pck[np.isnan(p)] = np.nan
                    pck *= sw

                    ck[k] = np.nansum(pck)
                    s[k] = np.nansum(p)
                    if s[k] == 0 or ck[k] < 0.25:
                        s[k] = np.nan
                    else:
                        s[k] /= ck[k]

            if np.all(np.isnan(s)):
                flag = -1
                logger.warn("Aperture is empty (probably outside the cube)")
                if not silent:
                    warnings.warn("Aperture is empty (probably outside the cube)")
                return s, region, mask, flag
            if np.sum(np.isnan(s)) / len(s) > 0.05:
                flag = -2
                logger.warn("A lot of NaNs in the spectrum (close to an edge)")
                if not silent:
                    warnings.warn("A lot of NaNs in the spectrum (close to an edge)")
                return s, region, mask, flag

            if useapprox:
                mask[ix] = 1
            else:
                mask = weight

        else:
            raise NotImplementedError("Unsupported pixco format for extract aperture")
            
        # --------------------------------------------------------------
        # SKY ANNULUS SUBTRACTION
        # --------------------------------------------------------------
        if sky_annulus is not None:
            sky_spec = np.zeros(shape[2])
            for k in range(shape[2]):
                wave = r["x"][2][k]  # wavelength in microns

                # --- choose aperture radius ---
                if rad == "auto":
                    rad_arcsec = psfranger(wave,scale, mode="MRSFWHMLaw2023")
                else:
                    rad_arcsec = float(rad)   # user-specified arcsec

                ann_scale = sky_annulus
                psize = r["cdelt"][1] * 3600
                rin = rad_arcsec * ann_scale
                rout = rad_arcsec * (ann_scale + 1)


                sky_w = annulus_weight(xx, yy, 0, 0, psize, rin, rout)
                if exclude_sectors is not None:
                    sector_w = sector_mask(xx, yy, 0, 0, exclude_sectors)

                    # Combine annulus + sector mask
                    sky_w = sky_w * sector_w
                
                
                
                p = data[:, :, k]
                w = sky_w


                sky_pixels = p[w > 0 & np.isfinite(p)]
                mean, med, std = sigma_clipped_stats(sky_pixels, sigma=3)
                sky_spec[k] = med


                if np.isnan(sky_spec[k]) or len(sky_pixels) < 10:
                    sky_spec[k] = np.nan

            #show where the weighted annulus is being applied
            fig, ax = plt.subplots(figsize=(8,8))
            ax.imshow(p, origin='lower', cmap='gray')

            # Annulus boundary
            ax.contour(sky_w > 0, levels=[0.5], colors='black', linewidths=2)

            # Filled annulus
            ann = np.zeros_like(p)
            ann[sky_w > 0] = 1
            ax.imshow(ann, origin='lower', alpha=0.3, cmap='Blues')
            ax.set_title("Annulus with Sector Mask")
            plt.show()
            # subtract sky
            s = s - sky_spec
    return s, region, mask, flag, omega   


def sigChans(xx, yy, noise, snr, thr):
    """
    Detect significant peaks in a sample series.
    
    function is used for catPeaks

    Parameters
    ----------
    xx : array-like 
        independent variable (e.g., wavelength or time).
    yy : array-like 
        signal values.
    noise : float 
        estimated 1-sigma noise level.
    snr : float 
        signal-to-noise threshold.
    thr : int 
        minimum separation (in samples) between peaks

    Returns
    ------------
    pp: Nx2 numpy array of peak positions and their amplitudes
    """
    import numpy as np
    from scipy.signal import find_peaks
    
    pp = np.zeros((1000, 2))  # preallocate
    ix = np.where(yy > snr * noise)[0]
    if len(ix) == 0:
        return np.empty((0, 2))

    dix = np.diff(ix) # difference between significant channel positions
    ul = np.append(np.where(dix > thr)[0] + 1, len(ix))  # chunk breaks
    npk = 0

    for j in range(len(ul)):
        if j == 0:
            zz = np.arange(ul[j])
        else:
            zz = np.arange(ul[j-1], ul[j])

        st = max(ix[zz[0]] - 1, 0)
        en = min(ix[zz[-1]] + 1, len(xx) - 1)
        xl = xx[st:en + 1]
        yl = yy[st:en + 1]

        if len(xl) == 2:
            k = np.argmax(yl)
            xc = xl[k]
            ixc = np.where(xx == xc)[0][0]

            if ixc <= 1 or ixc >= len(xx) - 2:
                pp[npk] = [xc, yl[k]]
                npk += 1
                continue

            xp = xx[ixc - 2:ixc + 3]
            yp = yy[ixc - 2:ixc + 3]
            p = np.polyfit(xp, yp, 2)

            if p[0] > 0:
                pp[npk] = [xc, yl[k]]
            else:
                x_peak = -p[1] / (2 * p[0])
                y_peak = p[2] - p[1] ** 2 / (4 * p[0])
                pp[npk] = [x_peak, y_peak]

            npk += 1
        else:
            peaks, _ = find_peaks(yl, prominence=noise / 2)
            if len(peaks) == 0:
                k = np.argmax(yl)
                peaks = np.array([k])

            for n, k in enumerate(peaks):
                xc = xl[k]
                ixc = np.where(xx == xc)[0][0]

                if ixc == 0 or ixc == len(xx) - 1:
                    pp[npk] = [xc, yl[k]]
                    npk += 1
                    continue

                xp = xx[ixc - 1:ixc + 2]
                yp = yy[ixc - 1:ixc + 2]
                p = np.polyfit(xp, yp, 2)

                if p[0] > 0:
                    pp[npk] = [xc, yl[k]]
                else:
                    x_peak = -p[1] / (2 * p[0])
                    y_peak = p[2] - p[1] ** 2 / (4 * p[0])
                    pp[npk] = [x_peak, y_peak]

                npk += 1

    return pp[:npk]




def fitLines(wv, spec, peaks, bounds, showfit):
    '''
    Goes through peak list and fits gaussians to them
    removes continuum. produces array with the gaussian
    solutions, another with the model, and another
    with the continuum
    
    function is used in catpeaks
    '''
    
    import numpy as np
    from scipy.signal import firwin, filtfilt
    from scipy.interpolate import interp1d
    
    c = 3e8  # speed of light (m/s)
    peaklist = peaks[:, 0]
    peakamp = peaks[:, 1]
    include = np.arange(len(wv))
    mask = np.zeros(len(wv), dtype=int)
    cont = [[] for _ in range(len(wv))]
    nl = len(peaklist)

    q = {
        'K': np.full(nl, np.nan),
        'S': np.full(nl, np.nan),
        'X0': np.full(nl, np.nan),
        'area': np.full(nl, np.nan),
        'eK': np.full(nl, np.nan),
        'eS': np.full(nl, np.nan),
        'eX0': np.full(nl, np.nan),
        'earea': np.full(nl, np.nan),
        'flag': -np.ones(nl, dtype=int)
    }

    # Dynamic mask width
    maskw = 3 + np.round(np.abs(np.log10(peakamp)) - 1) * 3
    maskw[maskw == 0] = 3
    maskw = maskw.astype(int)

    for i in range(nl):
        ix = np.argmin(np.abs(peaklist[i] - wv))
        st = max(ix - maskw[i], 0)
        en = min(ix + maskw[i] + 1, len(wv))
        mask[st:en] += 1
        for j in range(st, en):
            cont[j].append(i)

    include = include[mask == 0]
    base_interp = interp1d(wv[include], spec[include], kind="linear", fill_value="extrapolate")
    base = base_interp(wv)
    base[np.isnan(base)] = spec[np.isnan(base)]

    if np.min(wv) > 4:
        B = firwin(21, 0.005)
    else:
        B = firwin(21, 0.25)

    try:
        smo = filtfilt(B, [1], base)
    except Exception:
        return q, np.zeros_like(spec), np.zeros_like(spec)

    lines = spec - smo
    rms = np.std(lines[mask == 0])
    fitreg = 15
    ymodel = np.zeros_like(spec)

    for i in range(nl):
        ix = np.argmin(np.abs(peaklist[i] - wv))
        st = max(ix - fitreg, 0)
        en = min(ix + fitreg + 1, len(wv))
        wvreg = wv[st:en]
        spreg = lines[st:en]
        contrib = np.unique([j for sublist in cont[st:en] for j in sublist])
        X0 = peaklist[contrib]
        S = X0 / 1000 / 2.35
        K = np.zeros(len(contrib))
        for j, ci in enumerate(contrib):
            ix2 = np.argmin(np.abs(wvreg - peaklist[ci]))
            K[j] = spreg[ix2]

        if len(contrib) > 1:
            S /= 2
        
        if bounds != None:
            boundary = ([K*bounds[0][0], S*bounds[0][1], X0*bounds[0][2]] , [K*bounds[1][0], S*bounds[1][1], X0*bounds[1][2]])
        else:
            boundary = bounds
        Ko, So, X0o, chi2, yz, err, epar = gaussmfit(wvreg, spreg, K, S, X0, bounds=boundary, tol = 1e-10, showfit=showfit)
       
        
        ix_self = np.where(contrib == i)[0][0]
        q['K'][i] = Ko[ix_self]
        q['eK'][i] = epar[ix_self * 3]
        q['S'][i] = So[ix_self]
        q['eS'][i] = epar[ix_self * 3 + 1]
        q['X0'][i] = X0o[ix_self]
        q['eX0'][i] = epar[ix_self * 3 + 2]

        nu = c / (1e-6 * q['X0'][i])
        dnu = nu * q['S'][i] / q['X0'][i]
        #ednu =  np.sqrt(
         #   (c**2 / (1e-6 * q['X0'][i])**4) * (1e-6 * q['eS'][i])**2 +
         #   (c**2 * 4 * (1e-6 * q['S'][i])**2 / (1e-6 * q['X0'][i])**6) * (1e-6 * q['eX0'][i]) q**2
        #)
        ednu =  (nu / q['X0'][i]) * np.sqrt( (q['eS'][i])**2 + 4 * ((q['S'][i]**2) / q['X0'][i]**2) * q['eX0'][i]**2 )

        q['area'][i] = 1e-20 * np.sqrt(2 * np.pi) * q['K'][i] * dnu
        q['earea'][i] = 1e-20 * np.sqrt(2 * np.pi) * np.sqrt(
            dnu**2 * q['eK'][i]**2 + q['K'][i]**2 * ednu**2
        )
        

        q['flag'][i] = err
        if abs(X0o[ix_self] - X0[ix_self]) > 2 * np.mean(np.diff(wvreg)):
            q['flag'][i] += 8
        if q['K'][i] < 0:
            q['flag'][i] += 16
        if np.sqrt(chi2) > 3 * rms:
            q['flag'][i] += 32

        ymodel += q['K'][i] * np.exp(-((wv - q['X0'][i]) / (np.sqrt(2) * q['S'][i]))**2)
        
        #IPython.core.debugger.set_trace()

    return q, ymodel, smo

######################################################################################

def catPeaks(fname, outname, pixco=None, scale=1.5, snr=5, thr=2, sky_annulus=None, exclude_sectors=None, nchunk=1, use_exten=True, bounds=None, showfit=False, verbose = False, savefig=True):
    '''
    meant to plot and mark peaks from a spectrum.
    creates catalog of significant peaks for a spectrum
    
    Parameters:
    ---------------
    fname : str 
        Name of input file. Typically fits file.
        .txt and .csv 1D spectra are also now accepted 
    outname : str 
        Name of output file. formats as .txt file.
    pixco : array or list like 
        If "pixco" is an array of 1 row and 1 column i.e. [X,Y] it will assume the first 
        component is a row number and the second a column number for the pixel in
        the data, and extract that spectrum to analyze. If it has the form
        ([X1 , X2, X3], [Y1, Y2, Y3]), it will assume that the first row is the list of 
        row numbers and the second the list of column numbers of the area to average over. 
        If it is a list with two strings and a float (i.e. [RA, Dec, rad]) it will extract a spectrum at 
        that RA,DEC (colon-separated sexagesimal) in an aperture of that radius in arcsec.
        For a conical aperture input 'auto' for the rad. Currently supported for MIRI MRS spectra
    scale : float
        The scale factor to multiply the FWHM from the conical aperture to account for full PSF. defaults to 1.5
    snr : float 
        Minimum signal to noise for significant detection. Defaults to 5.
    thr : float 
        minimum separation in channels between significant areas in spectrum. Defaults to 2.
    sky_annulus : float
        If None no sky annulus will be subtracted from the spectrum. when given a number (i.e. 2) 
        will creaate an annulus with inner radius of the number times the radius used for science aperture
        will extend to 1 + the number given (ex. input of 2 will have outer radius 3x the aperture radius.
    exclude_sectors : array or list like
        Will mask out sectors of an annulus as defined by user.
        sector inputs should be given as angles in degrees defined as (deg_min, deg_max)
        If None will not mask out any sectors (i.e. defaults to None)
    nchunk : int 
        The number of chunks to cut the spectrum into, it defaults to 1. 
        If it is a vector the first and second component will be the start-end
        wavelengths (um) of the analysis.
    use_exten : bool
        adds arguement to rfits routine as to where to pull the data. 
        Defaults to True and defaults to reading the first extension.
    bounds : array_like
        Array containing lists the lower and upper bounds for each parameter to be scaled by (e.g., [[K_min, S_min, X0_min],[K_max, S_max, X0_max]]) (Default: None)
        Inputs should be given as floats i.e. 1.5 for 1.5 * K etc.
    showfit : bool
        shows gaussian fit (True) or not (False) (Default: False
    verbose : bool
        Print list of peaks to logger (True) or not (False) (Default: False)
     savefig : bool
        Whether to save the figure to a file (True) or not (False) (Default: True)   
    
    -------
    Returns: .txt file containing peak information
             .txt file containing the extracted spectra
             .txt file containing the continuum of region
             .tex file containing the model of the spectra
             .png plot of spectra with peaks marked and labeled
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    import os
    
    if isinstance(fname, list):
        if np.ndim(fname) != 2:
            raise ValueError("expecting list in form of [wavelength, flux]")
        x = fname[0]
        y = fname[1]
        if len(x) != len(y):
            raise ValueError("wavelength and flux inputs must have the same length")
        region = '1D spectrum'
        r = {'bunit': 'arbitrary', 'x': [None, None, x]}  # dummy metadata
        flag = 0
    else:
    
        #test the extention of the input to see how to handle
        ext = os.path.splitext(fname)[1].lower()


        # for 1d spectrum input as a .txt or .csv file
        if ext in ['.txt', '.csv']:
            try:
                data = np.loadtxt(fname, comments='#', delimiter=',' if ext == '.csv' else None)
            except Exception as e:
                raise ValueError(f"Could not read {fname}: {e}")

            ncols = data.shape[1]
            if ncols < 2:
                raise ValueError("File must contain at least two columns: wavelength and flux.")
            #assume the data is formated as wavelength then flux
            x = data[:, 0]
            y = data[:, 1]
            err = data[:, 2] if ncols >= 3 else None

            r = {'bunit': 'arbitrary', 'x': [None, None, x]}  # dummy metadata
            region = '1D spectrum'
            flag = 0

        else:
            # Handle input
            if isinstance(fname, str):
                if use_exten == True:
                    #assumes fits cube has additional extension where data is housed
                    r = rfits(fname, 'xten1')
                else:
                    r = rfits(fname)
            else:
                r = fname

            mloop = False
            if thr < 0:
                thr = 2
                mloop = True

            # Handle pixco / snr overload
            if pixco is None:
                pixco = 0
            elif isinstance(pixco, (int, float)) and snr == 5:
                snr = pixco

            snr = snr / 1.5  # MAD conversion

            s, region, mask, flag, omega = extract_aperture(r, pixco, scale, sky_annulus, exclude_sectors)
            if flag == -1:
                return {'Aperture is completely empty'}
            if flag == -2:
                print("Warning: spectrum contains NaNs, continuing with valid channels only.")

            x = np.array(r['x'][2])[~np.isnan(s)]
            y = s[~np.isnan(s)]

    if isinstance(nchunk, (list, tuple, np.ndarray)) and len(nchunk) == 2:
        stw, enw = nchunk
        ix1 = np.argmin(np.abs(stw - x))
        ix2 = np.argmin(np.abs(enw - x))
        nchunk = 1
        use_ix = True
    else:
        chunk = len(y) // nchunk
        use_ix = False

    np_detected = 0
    onp = 1
    mf = np.array([1, 1, 0, -1, -1])
    pp = []
    q_result = {'S': [], 'K': [], 'X0': [], 'area': [], 'flux': [], 'omega': [], 'eK': [], 'eS': [], 'eX0': [], 'earea': [], 'eflux': [], 'flag': []}

    for i in range(nchunk):
        if use_ix:
            st, en = ix1, ix2
        else:
            st = i * chunk
            en = min((i + 1) * chunk, len(y))

        xx = x[st:en]
        yy = y[st:en]
        dy = np.diff(yy)
        dx = (xx[1:] + xx[:-1]) / 2
        cc = np.convolve(dy, -mf, mode='same')
        scc = np.median(np.abs(cc - np.median(cc))) / 0.6745  # MAD

        p = sigChans(dx, cc, scc, snr, thr)
        q, ymo, cont = fitLines(xx, yy, p, bounds, showfit)
        
        
        # add break line to see if what is happening with the fitter
        flux_chunk = np.zeros(len(q['area']))
        eflux_chunk = np.zeros(len(q['area']))
        omega_chunk =np.zeros(len(q['area']))
        for j in range(len(q['area'])):
            if np.isnan(q['area'][j]):
                flux_chunk[j] = np.nan
                eflux_chunk[j] = np.nan
                continue
            if pixco is not None:
                Omega_at_line = omega[np.argmin(np.abs(xx - q['X0'][j]))]
                omega_chunk[j] = Omega_at_line
                flux_chunk[j] = q['area'][j] * Omega_at_line   # W/m^2
                eflux_chunk[j] = q['earea'][j] * Omega_at_line
            else:
                omega_chunk[j] = -0
                flux_chunk[j] = -0
                eflux_chunk[j] = -0
        np_detected += len(p)
        pp.extend(p)
        if savefig == True:
            # plotting
            plt.figure(1)
            plt.clf()
            plt.plot(xx, yy)
            plt.yscale('log')
            ax = plt.axis()
            for j, (loc, amp) in enumerate(p):
                nearest = np.argmin(np.abs(loc - xx))
                label_y = yy[nearest] * 0.99
                plt.plot([loc, loc], [10**np.floor(np.log10(min(yy))), max(yy)], 'r-.')
                plt.text(loc, label_y, str(onp + j), ha='center', va='top', fontsize=14)
            plt.axis([xx[0], xx[-1], ax[2], ax[3]])
            plt.xlabel('Wavelength (um)')
            plt.ylabel(f'Flux ({r["bunit"]})')
            plt.savefig(f"{outname}-chunk{i+1}.png" if nchunk > 1 else f"{outname}.png")

        # Print peaks
        if verbose == True:
            for j, (loc, amp) in enumerate(p):
                print(f"{onp + j:3d}   {loc:.4f} um {amp:8.4f} {r['bunit']}")
        onp = np_detected + 1

        # Append fit info
        for k in range(len(q['S'])):
            q_result['S'].append(q['S'][k])
            q_result['K'].append(q['K'][k])
            q_result['X0'].append(q['X0'][k])
            q_result['area'].append(q['area'][k])
            q_result['flux'].append(flux_chunk[k])
            q_result['eflux'].append(eflux_chunk[k])
            q_result['eK'].append(q['eK'][k])
            q_result['eS'].append(q['eS'][k])
            q_result['eX0'].append(q['eX0'][k])
            q_result['earea'].append(q['earea'][k])
            q_result['flag'].append(q['flag'][k])
            q_result['omega'].append(omega_chunk[k])
    # Write output files
    s2f = np.sqrt(np.log(256))
    with open(f'{outname}-peaklist.txt', 'w') as fp:
        if isinstance(fname, str):
            fp.write(f"%% Cube: {fname}\n")
        else:
            fp.write(f"%% Region spaning: {np.min(x)} {np.max(x)}\n")
        fp.write(f"%% Region: {region}\n")
        if region == 'pixel':
            fp.write(f"pixel {pixco[0]}, {pixco[1]}\n")
        elif region == 'list':
            fp.write("list " + " ".join(map(str, pixco[0])) + "\n")
            fp.write("%%              " + " ".join(map(str, pixco[1])) + "\n")
        elif region == 'circle' and isinstance(pixco, list):
            fp.write(f"%% circle {pixco[0]},{pixco[1]},{pixco[2]}\n")
        else:
            fp.write("%%undefined\n")

        fp.write("%% num wave(um) peak(MJy/sr) GaussK(MJy/sr) GaussFWHM(um) GaussX(um) Intensity(W/m2/sr) Flux(W/m2) ApRadius(arcsec) SolidAngle(omega) eGaussK eGaussS eGaussX eIntensity eFlux flag\n")
        for i in range(np_detected):
            S_abs = abs(q_result['S'][i])
            Inten = q_result['area'][i]
            Flux = q_result['flux'][i]
            eFlux = q_result['eflux'][i]
            if pixco is not None:
                rad = pixco[2]
                if rad == "auto":
                    rad_arcsec = psfranger(pp[i][0],scale, mode="MRSFWHMLaw2023")
                else:
                    rad_arcsec = float(rad)   # user-specified arcsec
            else:
                rad_arcsec = -0

            eS = q_result['eS'][i] * s2f
            fp.write(f"{i+1:3d} {pp[i][0]:8.4f} {pp[i][1]:12.4f} {q_result['K'][i]:12.4f} "
                     f"{S_abs * s2f:8.4f} {q_result['X0'][i]:8.4f} {Inten:8.4g} {Flux:8.4g} {rad_arcsec:8.4g} "
                     f"{q_result['omega'][i]:8.4g} {q_result['eK'][i]:8.4g} {eS:8.4g} {q_result['eX0'][i]:8.4g} "
                     f"{q_result['earea'][i]:8.4g} {eFlux:8.4g} {q_result['flag'][i]:2d}\n")

    # Spectrum output
    np.savetxt(f'{outname}_spec.txt', np.column_stack((xx, yy)), header=f'{fname}')
    np.savetxt(f'{outname}_cont.txt', np.column_stack((xx, cont)), header=f'{fname}')
    np.savetxt(f'{outname}_model.txt', np.column_stack((xx, ymo)), header=f'{fname}')

    return q_result

###################################################
#########Commands to manipulate line list##########
def recomputeLines(file,showfit):
    """
    Will recompute the peak list, the continuum, and, model
        based on new spec list i.e. when lines are cut redo model

    Parameters
    ----------
    file : str
        Base filename (without extension).
    
    returns
    -------
    edited lists in .txt format
    """
    import numpy as np
    import os
    # --- Select peaklist file ---
    if os.path.exists(f"{file}-peaklist-ed.txt"):
        fname = f"{file}-peaklist-ed.txt"
    else:
        fname = f"{file}-peaklist.txt"

    # --- Read header lines ---
    nh = 0
    hl = []
    with open(fname, "r") as fpr:
        while True:
            pos = fpr.tell()
            ln = fpr.readline()
            if not ln:
                break
            if ln.startswith("%"):
                nh += 1
                hl.append(ln.strip())
            else:
                fpr.seek(pos)
                break

    # --- Skip headers and inspect format ---
    with open(fname, 'r') as fp:
        cols = np.loadtxt(fp, comments='%', ndmin=2)
        C = [cols[:, i] for i in range(cols.shape[1])] 
    
    # --- Load spectrum ---
    u = np.loadtxt(f"{file}_spec.txt")

    # --- Fit lines ---
    q, ymo, cont = fitLines(u[:, 0], u[:, 1], np.column_stack([C[1], C[2]]), showfit)

    # --- Derived quantities ---
    s2f = np.sqrt(np.log(256))
    c = 3e8
    nu = c / (1e-6 * q["X0"])
    dnu = nu * q["S"] / q["X0"]
    #ednu = np.sqrt((q["S"]**2 / q["X0"]**4) * (q["eS"]**2) + old wrong error propagation
               #(4 * q["S"]**2 / q["X0"]**6) * (q["eX0"]**2))
    
    ednu =  (nu / q['X0']) * np.sqrt( (q['eS'])**2 + 4 * ((q['S']**2) / q['X0']**2) * q['eX0']**2 )
    area = 1e-20 * np.sqrt(2 * np.pi) * q["K"] * dnu
    earea = 1e-20 * np.sqrt(2 * np.pi) * np.sqrt((dnu**2) * (q["eK"]**2) +
                                             (q["K"]**2) * (ednu**2))


    Flux = area * C[9]
    eFlux = earea * C[9]
    # Combine output table
    K = np.column_stack([
        C[0].astype(int),     # ID
        C[1],                 # col2
        C[2],                 # col3
        q['K'],
        q['S'] * s2f,
        q['X0'],
        area,
        Flux,
        C[8],
        C[9],               #solid angle omega
        q['eK'],
        q['eS'] * s2f,
        q['eX0'],
        earea,
        eFlux,
        q['flag'].astype(int)
    ])

    # --- Rewrite peaklist file ---
    with open(fname, "w") as fpw:
        for h in hl:
            fpw.write(f"{h}\n")
        for row in K:
            fpw.write((
                f"{int(row[0]):3d} {row[1]:8.4f} {row[2]:12.4f} {row[3]:12.4f} "
                f"{row[4]:8.4f} {row[5]:8.4f} {row[6]:8.4g} {row[7]:8.4g} {row[8]:8.4g} {row[9]:8.4g} "
                f"{row[10]:8.4g} {row[11]:8.4g} {row[12]:8.4g} {row[13]:8.4g} {row[14]:8.4g} {int(row[15]):2d}\n"
            ))

    # --- Rewrite cont file ---
    contfile = f"{file}_cont.txt"
    hdr = []
    with open(contfile, "r") as fpr:
        for ln in fpr:
            if ln.startswith("%"):
                hdr.append(ln)
            else:
                break
    with open(contfile, "w") as fpw:
        fpw.writelines(hdr)
        for x, y in zip(u[:, 0], cont):
            fpw.write(f"{x:8.5f}  {y:8.5e}\n")

    # --- Rewrite model file ---
    modelfile = f"{file}_model.txt"
    hdr = []
    with open(modelfile, "r") as fpr:
        for ln in fpr:
            if ln.startswith("%"):
                hdr.append(ln)
            else:
                break
    with open(modelfile, "w") as fpw:
        fpw.writelines(hdr)
        for x, y in zip(u[:, 0], ymo):
            fpw.write(f"{x:8.5f}  {y:8.5e}\n")
            
    return



def keepLines(file, wave_list, tol=0.05, showfit=False):
    """
    Keep lines near a list of wavelengths.

    Parameters
    ----------
    file : str
        Base filename (without -peaklist suffix).
    wave_list : array_like
        List of wavelengths (um) to keep.
    tol : float or array_like, optional
        Tolerance(s) in um around each wavelength.
        If scalar, applied to all. If array, must match wave_list.
    showfit : bool
        Set to true to view the gaussian fitting of lines. Defaults to False
    
    returns
    -------
    peaklist with lines near specified wavelength(s).
    """
    import numpy as np

    fname = file + "-peaklist.txt"

     # --- Step 1: detect number of columns
    with open(fname, "r") as fp:
        fp.readline()   # skip % Cube
        fp.readline()   # skip % Region
        ln = fp.readline()  # header with column names

    
        ncols = 16
    
        

    # --- Step 2: load numeric data
    cols = np.loadtxt(
        fname,
        comments="%",
        usecols=range(ncols),
        unpack=True
        
    )

    # --- Normalize tol ---
    wave_list = np.atleast_1d(wave_list)
    tol = np.atleast_1d(tol)
    if tol.size == 1:
        tol = np.full_like(wave_list, tol[0], dtype=float)
    if tol.size != wave_list.size:
        raise ValueError("tol must be scalar or same length as wave_list")

    # --- Find rows within tolerance ---
    keep_idx = []
    for w, t in zip(wave_list, tol):
        match = np.where((cols[:, 1] >= (w - t)) & (cols[:, 1] <= (w + t)))[0]
        keep_idx.extend(match.tolist())
    keep_idx = np.unique(keep_idx)

    kept = cols[keep_idx, :]

    # --- Write output file ---
    outname = file + "-peaklist-ed.txt"
    with open(outname, "w") as f:
        for h in header:
            f.write(h + "\n")
        for row in kept:
            if ncols == 3:
                f.write(f"{int(row[0]):3d} {row[1]:8.4f} {row[2]:12.4f}\n")
            else:  # full Gaussian (12 cols)
                f.write(
                     f"{int(row[0]):3d} {row[1]:8.4f} {row[2]:12.4f} {row[3]:12.4f} "
                     f"{row[4]:8.4f} {row[5]:8.4f} {row[6]:8.4g} {row[7]:8.4g} {row[8]:8.4g} {row[9]:8.4g} "
                     f"{row[10]:8.4g} {row[11]:8.4g} {row[12]:8.4g} {row[13]:8.4g} {row[14]:8.4g} {int(row[15]):2d}\n"
                )

    
    print(f'Written {outname}')
    recomputeLines(file,showfit)
    return




def cullLines(file, remove_list, showfit=False):
    """
    Remove specific line indices from a peaklist and force creation of
    an '-ed' peaklist file.

    Parameters
    ----------
    file : str
        Base filename (without -peaklist suffix).
    remove_list : array_like
        List of line indices to remove.
    showfit : bool
        Set to true to view the gaussian fitting of lines. Defaults to False
    
    returns
    -------
    peaklist with specified lines removed
    """
    import numpy as np

    fname = file + "-peaklist.txt"

    # --- Step 1: detect number of columns
    with open(fname, "r") as fp:
        fp.readline()   # skip % Cube
        fp.readline()   # skip % Region
        ln = fp.readline()  # header with column names

    
        ncols = 16
    
        

    # --- Step 2: load numeric data
    cols = np.loadtxt(
        fname,
        comments="%",
        usecols=range(ncols),
        unpack=True
    )

    # --- Step 3: remove lines
    mask = ~np.isin(cols[0], remove_list)
    cols = [c[mask] for c in cols]

    # --- Step 4: write new file with same header lines
    outname = file + "-peaklist-ed.txt"
    with open(fname, "r") as fp:
        header = [line for line in fp if line.startswith("%")]

    with open(outname, "w") as f:
        for line in header:
            f.write(line)
        np.savetxt(f, np.column_stack(cols), fmt="%-12.6g")

    print(f"Written {outname} with {len(cols[0])} lines kept")
    recomputeLines(file,showfit)
    return




####################################################
#####Line-ID Helpers#################################
def parseCsvTable(fname):
    """
    Parse a simple CSV table with columns:
      LineName, RestValue, (optionally) transition.
    Returns: dict of arrays
    """
    import pandas as pd
    df = pd.read_csv(fname)
    data = {
        "LineName": df["Line Name"].astype(str).values,
        "RestValue": df["Rest Value"].astype(float).values
    }
    if "transition" in df.columns:
        data["transition"] = df["transition"].astype(str).values
    return data



def readPDR4all(fname):
    """
    Parse .lst file with fixed-width format from PDR4all.
    """
    import pandas as pd

    colspecs = [
        (0, 6),     # Model
        (7, 24),    # Wavelength (vacuum, micron)
        (25, 33),   # Species
        (34, 74),   # Transition
        (75, 84),   # A-coefficient
        (85, 96),   # Upper energy
        (97, 147),  # Comments
    ]
    names = ["Model", "RestValue", "Species", "transition",
             "A", "Eupper", "Comments"]

    df = pd.read_fwf(fname, colspecs=colspecs, names=names, skiprows=1)

    # Convert RestValue safely
        # Convert RestValue safely
    df["RestValue_raw"] = df["RestValue"]

    # Strip trailing ":" or other non-numeric junk before conversion
    df["RestValue_clean"] = df["RestValue_raw"].str.replace(":", "", regex=False).str.strip()

    df["RestValue"] = pd.to_numeric(df["RestValue_clean"], errors="coerce")


    # Warn if rows dropped
    dropped = df[df["RestValue"].isna()]
    if not dropped.empty:
        print(f"[readPDR4all] Dropped {len(dropped)} rows with invalid RestValue in {fname}:")
        for _, row in dropped.iterrows():
            print(f"   Species={row['Species']} | Transition={row['transition']} | Raw='{row['RestValue_raw']}'")

    # Keep only valid ones
    df = df.dropna(subset=["RestValue"])

    return {
        "LineName": df["Species"].astype(str).tolist(),
        "RestValue": df["RestValue"].astype(float).to_numpy(),
        "transition": df["transition"].astype(str).tolist()
    }


def readISO(fname):
    '''
    Helper for reading list of fine structure lines in the 2-205 micron range
    '''
    import pandas as pd
    df = pd.read_csv(
        fname,
        delim_whitespace=True,
        comment="#",
        names=["LineName","transition","RestValue","elambda","ep","ip"],
        skiprows=4
    )
    return {
        "LineName": df["LineName"].astype(str).tolist(),
        "transition": df["transition"].astype(str).tolist(),
        "RestValue": df["RestValue"].astype(float).values
    }

def readNIST(fname):
    '''
    code to read the NIST atomic line database in HTML
    '''
    import os 
    import numpy as np
    try:
        with open(fname, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
    except UnicodeDecodeError:
        with open(fname, "r", encoding="latin-1") as fp:
            lines = fp.readlines()


    # Skip the first 5 header lines
    lines = lines[5:]

    data = []
    for line in lines:
        if "|" not in line:
            continue
        parts = [p.strip() for p in line.split("|")]
        if len(parts) < 12:
            continue
        data.append(parts)

    # Convert to dictionary of lists
    LineName = []
    RestValue = []
    transition = []

    for row in data:
        name = row[0].replace(" ", "")
        rest = None
        try:
            rest = float(row[1])
        except:
            continue
        acc = float(row[2]) if row[2].replace(".","",1).isdigit() else 0.0

        # if accuracy < 1, mark as forbidden line
        if acc < 1:
            name = f"[{name}]"

        trans = f"{row[9].strip()}{row[10].strip()} - {row[6].strip()}{row[7].strip()}"
        LineName.append(name)
        RestValue.append(rest)
        transition.append(trans)

    return {"LineName": np.array(LineName),
            "RestValue": np.array(RestValue, dtype=float),
            "transition": np.array(transition)}



def _print_catalog_names(ignore_cats=None):
    ''' Helper function that prints the possible catalog names and brief info to the terminal.
    Helpful for the user to decide which (if any) catalogs to exclude.
    Parameters
    ----------
    ignore_cats : str, list of str, or None
        if not None, don't print the information about the listed catalogs (Default: None)
    '''
    import os

    script_dir = os.path.dirname(os.path.abspath(__file__))
    catdir = os.path.join(script_dir, "line_lists")

    catnames = [
        "Atomic-Ionic_FineStructure.csv", "Atomic-Ionic.csv", "H-He.csv", "H2.csv",
        "HeI-HeII.csv", "CO.csv", "publis20240406.lst",
        "ISOLineList.txt", "NIST-ASDLines.html"]

    if ignore_cats != None:
        for cname in ignore_cats:
            try:
                catnames.remove(cname)
            except ValueError:
                logger.error("Invalid catalog name %s in 'ignore_cats'. Proceeding without removing.")

    print('Available line catalogs are: '+', '.join(catnames))
    return


#####################################################################################
#####LineID code ####################################################################
def lineID(linefile, vlsr, R=250, outname=None, cat_path=None, ignore_cats=None, verbose=False):
    '''
    Line identifier that goes through peaks extracted from
    CatPeaks and identifies likely lines corresponding to them
    
    Parameters
    ----------------
    linefile : str 
        basepath to the peaklist from CatPeaks
        same path as used for outpath the user defined in CatPeaks.
    vlsr : float 
        velocity with respect to local standard of rest with units of km/s. 
        If input <= 20 assumes to be a redshift input instead. 
    R : float 
        precision level for uncertainty.
    outname : str 
        desired path to outfile if 
        new destination is desired. Optional. 
        Defaults to linefile-lineID.txt if none.
    cat_path: str 
        optional path to line list 
        if lists are not in the same directory.
    ignore_cats : list or None
        Names of catalogs to ignore (Default: None)
    verbose : bool
        Print list of peaks to logger (True) or not (False) (Default: False)
        
    Returns:
    .txt file containing information on likely identifed lines given characteristics.
    ---------------
    
    '''
    
    import os 
    import numpy as np
    clight = 3e5
    # Handle file or array input
    if isinstance(linefile, str):
        if outname is None:
            outname = f"{linefile}-lineID.txt"
        if os.path.exists(f"{linefile}-peaklist-ed.txt"):
            fname = f"{linefile}-peaklist-ed.txt"
        else:
            fname = f"{linefile}-peaklist.txt"
        with open(fname, "r") as fp:
            fp.readline(); fp.readline()
            ln = fp.readline()
            if "Gauss" in ln:
                cols = np.loadtxt(fname, comments="%", skiprows=3, usecols=range(16), unpack=True)
            else:
                cols = np.loadtxt(fname, comments="%", skiprows=3, usecols=range(3), unpack=True)
        lineno, wv = cols[0].astype(int), cols[1]
    elif isinstance(linefile, (list, np.ndarray)):
        wv = np.array(linefile)
        lineno = np.arange(1, len(wv) + 1)
        linefile = "default"
        if outname is None:
            outname = f"{linefile}-lineID.txt"
    else:
        raise ValueError("First parameter type not supported.")
    
    #ensure lineno and wv are arrays
    lineno, wv = np.atleast_1d(lineno), np.atleast_1d(wv)
    
    # open output
    fp = open(outname, "w")

    if R is None:
        R = 250
        print(f"Adopting default precision lambda/Dlambda={R}")

    # Catalogs
    
    # --- Determine catalog directory ---
    if cat_path is not None:
        # use the user-provided one
        catdir = cat_path
    else:
        # default to Line_lists next to this script
        script_dir = os.path.dirname(os.path.abspath(__file__))
        catdir = os.path.join(script_dir, "Line_lists")

    # --- Verify it exists ---
    if not os.path.exists(catdir):
        raise ValueError(f"Path to line lists not found: {catdir}")
    catnames = [
        "Atomic-Ionic_FineStructure.csv", "Atomic-Ionic.csv", "H-He.csv", "H2.csv",
        "HeI-HeII.csv", "CO.csv", "publis20240406.lst",
        "ISOLineList.txt", "NIST-ASDLines.html"
    ]
    
    if ignore_cats != None:
        for cname in ignore_cats:
            try:
                catnames.remove(cname)
                logger.info("Removing %s from catalog list" %catnames)
            except ValueError:
                logger.error("Invalid catalog name %s in 'ignore_cats'. Proceeding without removing.")

    cat = []
    logger.info("Checking for line candidates against "+", ".join(str(cname) for cname in catnames))
    for cname in catnames:
        if ".csv" in cname:
            cat.append(parseCsvTable(os.path.join(catdir, cname)))
        elif ".lst" in cname:
            cat.append(readPDR4all(os.path.join(catdir, cname)))
        elif "ISOLineList" in cname:
            cat.append(readISO(os.path.join(catdir, cname)))
        elif "NIST" in cname:
            cat.append(readNIST(os.path.join(catdir, cname)))
        else:
            raise ValueError(f"Catalog {cname} not recognized.")

    # velocity or z
    if vlsr > 20:
        wvr = wv / (1 + vlsr/clight)
        fp.write(f"Matching lines in {linefile} assuming vlsr={vlsr:.0f} km/s\n")
        print(f"Matching lines in {linefile} assuming vlsr={vlsr:.0f} km/s")
    else:
        wvr = wv / (1 + vlsr)
        fp.write(f"Matching lines in {linefile} assuming redshift z={vlsr:.5f}\n")
        print(f"Matching lines in {linefile} assuming redshift z={vlsr:.5f}")
    fp.write("\n")

    nm = 0
    for i, (lnum, w) in enumerate(zip(lineno, wv)):
        err = w / R
        fp.write(f"#{lnum:3d} {w:8.4f}+/-{err:0.4f} um (rest={wvr[i]:8.4f} um):\n       ")
        if verbose == True:
            logger.info(f"#{lnum:3d} {w:8.4f}+/-{err:0.4f} um (rest={wvr[i]:8.4f} um):\n       ")

        dd, tn, sp, rv, cn = [], [], [], [], []
        nl = 0
        for j, catalog in enumerate(cat):
            dist = np.abs(wvr[i] - catalog["RestValue"])
            ix = np.where(dist <= w/R)[0]
            if len(ix) > 0:
                for k in ix:
                    if "transition" in catalog:
                        name = catalog["LineName"][k].replace(" ", "") + " " + catalog["transition"][k]
                    else:
                        name = catalog["LineName"][k]
                    rv.append(catalog["RestValue"][k])
                    dd.append(wvr[i] - rv[-1])
                    tn.append(name)
                    sp.append(catalog["LineName"][k].replace(" ", ""))
                    cn.append(j)
                    nl += 1

        if nl > 0:
            ix = np.argsort(np.abs(dd))
            # move H2 vibrational transitions to bottom
            rit = 0
            while rit < len(ix) and "H2 v=" in tn[ix[0]]:
                ix = np.roll(ix, -1)
                rit += 1

            for jdx, idx in enumerate(ix):
                velshift = dd[idx] / rv[idx] * clight
                if jdx == 0:
                    fp.write(f"| {tn[idx]} {velshift:+4.0f} | ")
                    if verbose == True:
                        logger.info(f"| {tn[idx]} {velshift:+4.0f} | ", end="")
                else:
                    fp.write(f"{tn[idx]} {velshift:+4.0f} | ")
                    if verbose == True:
                        logger.info(f"{tn[idx]} {velshift:+4.0f} | ", end="")
            fp.write("\n\n")

            fp.write(f"       BEST: DLambda={dd[ix[0]]:0.4f} um, Catalog={catnames[cn[ix[0]]]}, Restwv={rv[ix[0]]:0.4f}\n       {tn[ix[0]]}\n")
            if verbose == True:
                logger.info(f"       BEST: DLambda={dd[ix[0]]:0.4f} um, Catalog={catnames[cn[ix[0]]]}, Restwv={rv[ix[0]]:0.4f}\n       {tn[ix[0]]}")
            if len(ix) > 1:
                k = 1
                found = True
                while abs(dd[ix[0]] - dd[ix[k]]) < 1e-4:
                    if k+1 < len(ix):
                        k += 1
                    else:
                        found = False
                        break
                if found:
                    fp.write(f"       NEXT: DLambda={dd[ix[k]]:0.4f} um, Catalog={catnames[cn[ix[k]]]}, Restwv={rv[ix[k]]:0.4f}\n       {tn[ix[k]]}\n")
                    if verbose == True:
                        logger.info(f"       NEXT: DLambda={dd[ix[k]]:0.4f} um, Catalog={catnames[cn[ix[k]]]}, Restwv={rv[ix[k]]:0.4f}\n       {tn[ix[k]]}")
        else:
            fp.write(f"       NOMATCH within +/-{w/R:0.4f} um\n")
            if verbose == True:
                logger.info(f"       NOMATCH within +/-{w/R:0.4f} um")
            nm += 1

    fp.close()
    logger.info(f"Attempted IDs for {len(wv)} lines, found no matches for {nm} lines")
    return

######## Additional commands ####################

def findLines(fname, outname, wmin, wmax, pixco=None, sky_annulus=None, scale=1.5, snr=5, thr=2, showfit=False, ID=False, vlsr=None, catdir=None):
    """
    Extract spectrum from FITS cube and identify peaks within a wavelength range.
    Works like catPeaks, but restricted to [wmin, wmax]. output files are default labeled with 
    wavelength range.
    lineID may also be ran internally through code.
    
    

    Parameters
    ----------
    fname : str
        Input FITS cube filename.
    outname : str
        Base path name for output files.
    wmin, wmax : float
        Wavelength range in microns.
    pixco : list/tuple, optional
        Pixel/aperture specification (see catPeaks).
    scale : float
        The scale factor to multiply the FWHM from the conical aperture to account for full PSF. defaults to 1.5
    snr, thr : float
        Peak finding parameters.(see catPeaks)
    showfit : bool
        Set to true to view the gaussian fitting of lines. Defaults to False
    ID : Bool
        If user wants to perform lineID also set True.
    vlsr: float
        Line ID parameter (see linID).
    catdir: str
        path to line lists if not default path.

    Returns: .txt file containing peak information
             .txt file containing the extracted spectra
             .txt file containing the continuum of region
             .tex file containing the model of the spectra
             .png plot of spectra with peaks marked and labeled
        if ID is True
            .txt file containing information on likely identifed lines given characteristics
    ''''
    -------
    
    """
    import numpy as np
    import matplotlib.pyplot as plt

    # Handle input FITS
    if isinstance(fname, str):
        r = rfits(fname, 'xten1')
    else:
        r = fname

    if pixco is None:
        pixco = 0
    elif isinstance(pixco, (int, float)) and snr == 5:
        snr = pixco
    snr = snr / 1.5  # MAD conversion

    # Extract spectrum
    s, region, mask, flag, omega = extract_aperture(r, pixco, scale, sky_annulus)
    if flag != 0:
        return {'flag': flag}

    x = np.array(r['x'][2])[~np.isnan(s)]
    y = s[~np.isnan(s)]

    # Restrict to wavelength range
    ix1 = np.argmin(np.abs(wmin - x))
    ix2 = np.argmin(np.abs(wmax - x))
    xx = x[ix1:ix2+1]
    yy = y[ix1:ix2+1]

    # Peak finding
    mf = np.array([1, 1, 0, -1, -1])
    dy = np.diff(yy)
    dx = (xx[1:] + xx[:-1]) / 2
    cc = np.convolve(dy, -mf, mode='same')
    scc = np.median(np.abs(cc - np.median(cc))) / 0.6745
    p = sigChans(dx, cc, scc, snr, thr)

    q_result, ymo, cont = fitLines(xx, yy, p, showfit)
    
    flux_chunk = np.zeros(len(q_result['area']))
    eflux_chunk = np.zeros(len(q_result['area']))
    omega_chunk =np.zeros(len(q_result['area']))
    for j in range(len(q_result['area'])):
        if np.isnan(q_result['area'][j]):
            flux_chunk[j] = np.nan
            eflux_chunk[j] = np.nan
            continue

        Omega_at_line = omega[np.argmin(np.abs(xx - q_result['X0'][j]))]
        omega_chunk[j] = Omega_at_line
        flux_chunk[j] = q_result['area'][j] * Omega_at_line   # W/m^2
        eflux_chunk[j] = q_result['earea'][j] * Omega_at_line
    
    # Plot
    plt.figure(1)
    plt.clf()
    plt.plot(xx, yy)
    plt.yscale('log')
    ax = plt.axis()
    for j, (loc, amp) in enumerate(p):
        nearest = np.argmin(np.abs(loc - xx))
        label_y = yy[nearest] * 0.99
        plt.plot([loc, loc], [10**np.floor(np.log10(min(yy))), max(yy)], 'r-.')
        plt.text(loc, label_y, str(j+1), ha='center', va='top', fontsize=14)
    plt.axis([xx[0], xx[-1], ax[2], ax[3]])
    plt.xlabel('Wavelength (um)')
    plt.ylabel(f'Flux ({r["bunit"]})')
    plt.savefig(f"{outname}-findLines{wmin}-{wmax}mircon.png")

    # Write output files
    s2f = np.sqrt(np.log(256))
    with open(f'{outname}-findLines{wmin}-{wmax}mircon-peaklist.txt', 'w') as fp:
        fp.write(f"%% Cube: {fname}\n")
        fp.write(f"%% Region: {region}\n")
        fp.write(f"%% Wavelength range: {wmin:.3f}-{wmax:.3f} µm\n")
        fp.write("%% num wave(um) peak(MJy/sr) GaussK(MJy/sr) GaussFWHM(um) GaussX(um) Intensity(W/m2/sr) Flux(W/m2)" 
                 "ApRadius(arcsec) SolidAngle(omega) eGaussK eGaussS eGaussX eIntensity eFlux flag\n")
        for i in range(len(p)):
            S_abs = abs(q_result['S'][i])
            Inten = q_result['area'][i]
            Flux = flux_chunk[i]
            eFlux = eflux_chunk[i]
            omega = omega_chunk[i]
            rad = pixco[2]
            if rad == "auto":
                rad_arcsec = psfranger(p[i][0],scale, mode="MRSFWHMLaw2023")
            else:
                rad_arcsec = float(rad)   # user-specified arcsec

            eS = q_result['eS'][i] * s2f
            fp.write(f"{i+1:3d} {p[i][0]:8.4f} {p[i][1]:12.4f} {q_result['K'][i]:12.4f} "
                     f"{S_abs * s2f:8.4f} {q_result['X0'][i]:8.4f} {Inten:8.4g} {Flux:8.4g} {rad_arcsec:8.4g} "
                     f"{omega:8.4g} {q_result['eK'][i]:8.4g} {eS:8.4g} {q_result['eX0'][i]:8.4g} "
                     f"{q_result['earea'][i]:8.4g} {eFlux:8.4g} {q_result['flag'][i]:2d}\n")

    np.savetxt(f'{outname}_spec{wmin}-{wmax}mircon.txt', np.column_stack((xx, yy)), header=f'{fname}')
    np.savetxt(f'{outname}_cont{wmin}-{wmax}mircon.txt', np.column_stack((xx, cont)), header=f'{fname}')
    np.savetxt(f'{outname}_model{wmin}-{wmax}mircon.txt', np.column_stack((xx, ymo)), header=f'{fname}')

    print(f"findLines: detected {len(p)} lines in {wmin}-{wmax} µm")
    
    if ID==True:
        if vlsr is None:
            raise ValueError("Must define vlsr to perform Line identification")
        if catdir is None:
            lineID(f'{outname}-findLines{wmin}-{wmax}mircon',vlsr)
        else:
            lineID(f'{outname}-findLines{wmin}-{wmax}mircon',vlsr,cat_path=catdir)
        
    
    return 



def findSpecies(linefile, species_list, vlsr, R=250, transition_filter=None, cat_path=None, outname=None, plot=False):
    """
    Identify whether specified species appear in the detected spectrum.
    
    This code is still in beta and more prone to bugs

    Parameters
    ----------
    linefile : str
        Base filename (without extension), i.e. same as used in lineID.
    species_list : str or list of str
        Species names (or substrings) to search for, e.g. "CO" or ["CO","13CO"].
    vlsr : float
        Velocity (km/s) or redshift (if <=20, treated as z).
    R : float
        Resolving power (lambda/dlambda).
    transition_filter : str or list of str, optional
        Only include matches whose transition description contains this substring.
        Example: "v=0" to restrict to ground-vibrational H2 lines.
    cat_path : str, optional
        Path to catalog directory. Defaults to Line_lists next to this script.
    outname : str, optional
        Output filename. Defaults to "<linefile>-speciesID.txt".
    plot : bool
        If True, overlay identified species on spectrum plot.

    Returns
    -------
    matches : list of dict
        Each dict has keys: line_num, obs_wv, rest_wv, species, transition, velshift.
    """
    import os, numpy as np
    import matplotlib.pyplot as plt
    clight = 3e5

    if isinstance(species_list, str):
        species_list = [species_list]

    # --- Load observed peaklist ---
    if os.path.exists(f"{linefile}-peaklist-ed.txt"):
        fname = f"{linefile}-peaklist-ed.txt"
    else:
        fname = f"{linefile}-peaklist.txt"

    with open(fname, "r") as fp:
        fp.readline(); fp.readline()
        ln = fp.readline()
        if "Gauss" in ln:
            cols = np.loadtxt(fname, comments="%", skiprows=3, usecols=range(16), unpack=True)
        else:
            cols = np.loadtxt(fname, comments="%", skiprows=3, usecols=range(3), unpack=True)

    lineno, wv = cols[0].astype(int), cols[1]

    # --- Rest wavelength correction ---
    if vlsr > 20:
        wvr = wv / (1 + vlsr/clight)
        mode = f"vlsr={vlsr:.0f} km/s"
    else:
        wvr = wv / (1 + vlsr)
        mode = f"redshift z={vlsr:.5f}"

    # --- Determine catalog directory ---
    if cat_path is not None:
        catdir = cat_path
    else:
        script_dir = os.path.dirname(os.path.abspath(__file__))
        catdir = os.path.join(script_dir, "Line_lists")
    if not os.path.exists(catdir):
        raise ValueError(f"Path to line lists not found: {catdir}")

    catnames = [
        "Atomic-Ionic_FineStructure.csv", "Atomic-Ionic.csv", "H-He.csv", "H2.csv",
        "HeI-HeII.csv", "CO.csv", "publis20240406.lst",
        "ISOLineList.txt", "NIST-ASDLines.html"
    ]

    # --- Load catalogs ---
    catalogs = []
    for cname in catnames:
        path = os.path.join(catdir, cname)
        if ".csv" in cname:
            catalogs.append(parseCsvTable(path))
        elif ".lst" in cname:
            catalogs.append(readPDR4all(path))
        elif "ISOLineList" in cname:
            catalogs.append(readISO(path))
        elif "NIST" in cname:
            catalogs.append(readNIST(path))

    # --- Open output ---
    if outname is None:
        if transition_filter is None:
            outname = f"{linefile}-species{species_list}.txt"
        else:
            outname = f"{linefile}-species{species_list}{transition_filter}.txt"
    fp = open(outname, "w", encoding="utf-8")   # safer encoding
    fp.write(f"Species search: {species_list}, {mode}\n\n")

    matches = []
    for i, (lnum, w) in enumerate(zip(lineno, wvr)):
        err = w / R
        found = []
        for cat in catalogs:
            for j, rv in enumerate(cat["RestValue"]):
                if abs(w - rv) <= err:
                    name = cat["LineName"][j]
                    transition = cat.get("transition", [""]*len(cat["RestValue"]))[j]

                    if any(sp.lower() in name.lower() for sp in species_list):
                        # --- Apply transition filter if requested ---
                        if transition_filter:
                            if isinstance(transition_filter, str):
                                filters = [transition_filter]
                            else:
                                filters = transition_filter
                            # check both LineName and transition fields
                            if not any(tf.lower() in name.lower() or tf.lower() in transition.lower() 
                                       for tf in filters):
                                continue


                        velshift = (w - rv) / rv * clight
                        found.append((name, transition, rv, velshift))

        if found:
            fp.write(f"#{lnum:3d} {w:8.4f} +/-{err:.4f} um:\n")
            for name, t, rv, vel in found:
                fp.write(f"   {name} {t} | Rest={rv:.4f} um | dv={vel:+.1f} km/s\n")
                matches.append({
                    "line_num": lnum,
                    "obs_wv": w,
                    "rest_wv": rv,
                    "species": name,
                    "transition": t,
                    "velshift": vel
                })
            fp.write("\n")


    fp.close()
    print(f"findSpecies: wrote {len(matches)} matches to {outname}")

    # --- Optional plotting ---
    if plot and matches:
        try:
            specfile = f"{linefile}_spec.txt"
            data = np.loadtxt(specfile)
            xx, yy = data[:, 0], data[:, 1]

            plt.figure(figsize=(8, 5))
            plt.plot(xx, yy, color="k", lw=1, zorder=10)
            plt.yscale("log")
            plt.xlabel("Wavelength (um)")
            plt.ylabel(f"Intensity ({r['bunit']})")

            for m in matches:
                x = m["obs_wv"]
                plt.axvline(x, color="r", ls="--", alpha=0.6, zorder=5)
                label = f"{m['species']}"
                if m["transition"]:
                    label += f" {m['transition']}"
                plt.text(x-0.05, max(yy)*0.4, label, rotation=90,
                         ha="center", va="bottom", fontsize=8, color="red")

            plotname = outname.replace(".txt", ".png")
            plt.tight_layout()
            plt.savefig(plotname, dpi=150)
            plt.close()
            print(f"findSpecies: plot saved to {plotname}")
        except Exception as e:
            print(f"[findSpecies] Plotting failed: {e}")

    return matches


def readLineID(filename: str) -> Dict[str, Any]:
    """
    Parse a *-lineID.txt file.
    
    Parameters
    ----------
    filename : str
        .txt file containing the line id information
        if using defaults should be of form *-lineID.txt.

    Returns
    -------
    {
      "vlsr_kms": float or None,
      "blocks": [
         {
           "line_num": int,
           "obs_wv_um": float,
           "obs_err_um": float,
           "rest_header_um": float,   # the (rest= ... um) shown on the # line
           "candidates": [ {"label": str, "vel_kms": int}, ... ],
           "best": {"dlambda_um": float, "catalog": str, "restwv_um": float, "label": str} or None,
           "next": {"dlambda_um": float, "catalog": str, "restwv_um": float, "label": str} or None,
         },
         ...
      ]
    }
    """

    # Regex patterns
    hdr_vlsr_re = re.compile(r"assuming\s+vlsr\s*=\s*([\-0-9.]+)\s*km/s", re.I)
    block_re = re.compile(
        r"^#\s*(\d+)\s+([\-0-9.]+)\+/-([\-0-9.]+)\s*um\s*\(rest=\s*([\-0-9.]+)\s*um\):\s*$"
    )
    bestnext_re = re.compile(
        r"^(BEST|NEXT):\s*DLambda=([\-0-9.]+)\s*um,\s*Catalog=([^,]+),\s*Restwv=([\-0-9.]+)\s*$",
        re.I
    )
    # Candidate chunks look like: | <label ...> <+/-INT> |
    # We capture repeatedly on a line.
    cand_re = re.compile(r"\|\s*(.*?)\s+([+\-]?\d+)\s*(?=\|)")

    # Read file
    with open(filename, "r", encoding="utf-8", errors="replace") as f:
        lines = f.readlines()

    # VLSR (from the first line typically)
    vlsr_kms = None
    for l in lines[:5]:
        m = hdr_vlsr_re.search(l)
        if m:
            try:
                vlsr_kms = float(m.group(1))
            except Exception:
                vlsr_kms = None
            break

    blocks: List[Dict[str, Any]] = []
    i = 0
    n = len(lines)

    while i < n:
        line = lines[i]
        m = block_re.match(line.strip("\n"))
        if not m:
            i += 1
            continue

        # Parse the block header line
        line_num = int(m.group(1))
        obs_wv = float(m.group(2))
        obs_err = float(m.group(3))
        rest_header = float(m.group(4))

        i += 1
        # Collect candidate lines (may be one or more lines containing '|')
        candidates = []
        while i < n and 'BEST:' not in lines[i] and 'NEXT:' not in lines[i] and not lines[i].lstrip().startswith('#'):
            if '|' in lines[i]:
                for cm in cand_re.finditer(lines[i]):
                    label = cm.group(1).strip()
                    vel = int(cm.group(2))
                    candidates.append({"label": label, "vel_kms": vel})
            i += 1

        best = None
        nxt = None

        # Parse BEST (if present)
        if i < n and lines[i].lstrip().upper().startswith('BEST:'):
            bm = bestnext_re.match(lines[i].strip())
            if bm:
                dl, cat, rw = float(bm.group(2)), bm.group(3).strip(), float(bm.group(4))
                best_label = lines[i+1].strip() if i+1 < n else ""
                best = {"dlambda_um": dl, "catalog": cat, "restwv_um": rw, "label": best_label}
                i += 2
            else:
                i += 1  # fail-safe advance

        # Parse NEXT (if present)
        if i < n and lines[i].lstrip().upper().startswith('NEXT:'):
            nm = bestnext_re.match(lines[i].strip())
            if nm:
                dl, cat, rw = float(nm.group(2)), nm.group(3).strip(), float(nm.group(4))
                next_label = lines[i+1].strip() if i+1 < n else ""
                nxt = {"dlambda_um": dl, "catalog": cat, "restwv_um": rw, "label": next_label}
                i += 2
            else:
                i += 1

        blocks.append({
            "line_num": line_num,
            "obs_wv_um": obs_wv,
            "obs_err_um": obs_err,
            "rest_header_um": rest_header,
            "candidates": candidates,
            "best": best,
            "next": nxt
        })

    return {"vlsr_kms": vlsr_kms, "blocks": blocks}

##############################################################################
################### Plotting Routines ################################################

def plotID(linename, specname, savefig=False, outname=None):
    '''
    quick plotter that will plot the spectra 
    with labels for ID'd lines.
    lineID must be run before to have the correct files available
    
    
    Parameters
    ------------
    linename : str
        path to .txt file containing identified lines from lineID.
    specname : str
        path to .txt file containing the spectrum extracted from catPeaks.
    savefig : bool
        Tells function if the figure should be saved. defaults to False.
    outname : str or None
        desired path and file name for the plot. Must be defined if savefig=True.
    
    Returns
    -------
    plot of spectra with identified lines marked and labeled
    
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    basedata = np.loadtxt(specname)
    x, y = basedata[:, 0], basedata[:, 1]
    top = max(y)
    
    res = readLineID(linename)
    best_labels = [b["best"]["label"] for b in res["blocks"] if b["best"]]
    obwave = [k['obs_wv_um'] for k in res['blocks']]
    
    plt.figure(figsize=(9,8))
    plt.plot(x, y, color='k', lw=1, zorder=10)
    plt.yscale('log')
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Flux (MJy/sr)')
    
    for j in range(len(obwave)):
        plt.axvline(obwave[j], color='r', ls='--', alpha=0.6, zorder=5)
        plt.text(obwave[j]-0.05, top*0.5, best_labels[j], rotation=90,
                 ha='center', va='bottom', fontsize=8, color='red')
        
    if savefig is True:
        if outname is None:
            raise ValueError('Must define a path for figure to save to')
        plt.savefig(outname, bbox_inches='tight')
    
    plt.show()
    
    
    return

def plotIDClean(linename, specname, savefig=False, outname=None):
    '''
    quick plotter that will plot the spectra 
    with general labels for specific line types i.e. HI lines.
    lineID must be run before to have the correct files available
    
    
    Parameters
    ------------
    linename : str
        path to .txt file containing identified lines from lineID.
    specname : str
        path to .txt file containing the spectrum extracted from catPeaks.
    savefig : bool
        Tells function if the figure should be saved. defaults to False.
    outname : str or None
        desired path and file name for the plot. Must be defined if savefig=True.
    
    Returns
    -------
    plot of spectra with identified lines marked and labeled
    
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D 
    
    basedata = np.loadtxt(specname)
    x, y = basedata[:, 0], basedata[:, 1]
 
    
    res = readLineID(linename)
    best_labels = [b["best"]["label"] for b in res["blocks"] if b["best"]]
    obwave = [k['obs_wv_um'] for k in res['blocks']]
    
    
    # Define a mapping for line types to colors
    line_colors = {
        "HI": "cornflowerblue",         # Hydrogen lines (Balmer, Paschen, Brackett, etc.)
        "HeI": "orange",      # Helium lines
        "H2": "purple",        # H2 lines
    }
    default_color = "red"   # fallback color for anything not specified
    
    
    plt.figure(figsize=(11,3))
    plt.plot(x, y, color='k', lw=1, zorder=10)
    #plt.ylim(4e3, 4e4)
    plt.yscale('log')
    plt.xlabel('Wavelength (um)')
    plt.ylabel('Flux')
    
    for j, label in enumerate(best_labels):
        # Decide color based on label
        color = default_color
        for key, c in line_colors.items():
            if label.startswith(key):   # e.g. "Hα", "Hβ", "He I"
                color = c
                break

        plt.axvline(obwave[j], color=color, ls='--', alpha=0.6, zorder=5)
    return


def viewFit(file_path, wave_range=None, flux_range=None, scale_model=False, showline=None):
    '''
    Viewer to see the extracted spectra, the continuum fit, and the modeled gaussian profiles of peaks.
    Model is plotted below the spectra to show which peaks have had gaussians fit.
    
    Parameters:
    ------------
    file_path : str
        Primary path to where the spectra, model, and continuum data are held
        Assumes all three have the same name except for specific extensions grabbed when reading
    wave_range : array
        Wavelength range to use as an xlim given in the form [x,y] with x as the min and y as the max
        Defaults to None
    flux_range : array
        Flux range to use as a ylim given in the form [x,y] with x as the min and y as the max
        Defaults to None
    scale_model : bool
        Set to true if you want to display the model on top of the spectra (i.e. scales the model by the continuum)
        Defaults to False
    showline : float
        simple way to see where a particular line is by inputting its wavelength
        defaults to None
    
    ------------
    Returns:
        Plot of the spectra (in black), the model (blue), and the continuum (red)
    
    
    '''
    
    import numpy as np
    import matplotlib.pyplot as plt
    
    #get specific files
    contdata = file_path + '_cont.txt'
    modeldata = file_path + '_model.txt'
    specdata = file_path + '_spec.txt'
    
    contin = np.loadtxt(contdata)
    modeldat = np.loadtxt(modeldata)
    basedat = np.loadtxt(specdata)
    a, b = contin[:, 0], contin [:, 1]

    j, k = modeldat[:, 0], modeldat[:, 1]

    x, y = basedat[:, 0], basedat[:, 1]


    plt.figure(figsize=(9,8))
    plt.plot(x, y, color='k', lw=1, zorder=100, label = 'spectra')
    if scale_model is True:
        plt.plot(j, k+b, color = 'b', label ='model')
    else:
        plt.plot(j, k, color='b', label = 'model')
    plt.plot(a, b, color='r', zorder=9, label = 'continuum')
    if showline is not None:
        plt.axvline(showline)
    if wave_range is not None:
        if len(wave_range) != 2:
            raise ValueError(' wave_range must be given as [x,y] with x as the minimum and y as the maximum')
        plt.xlim(wave_range[0], wave_range[1])
    if flux_range is not None:
        if len(flux_range) != 2:
            raise ValueError(' flux_range must be given as [x,y] with x as the minimum and y as the maximum')
        plt.ylim(flux_range[0],flux_range[1])

    plt.xlabel('Wavelength (um)')
    plt.ylabel('Surface Brightness (MJy/sr)')
    plt.legend()
    plt.show()
    return

#######################################################################

#####################List editors ######################################

##########################################################################
def combList(path,specs, peak_list, line_list):
    '''
    Simple way to combine output lists into master lists. 
    All previous formatting is left in place

    Parameters
    ------------
    path : str
        The path you want the files to go towards
    specs : list of str
        The path to the spectrum files (typically _spec.txt)
    peak_list : list of str
        The root path to the peaklist file
        Will search for if an edited peaklist file exits
    line_list : list of str
        The path to the lineID files (typically -lineID.txt)
    
    Returns
    -----------
    combined .txt files
    '''
    
    
    with open(path +'allCH_spec.txt', 'w') as outfile:
        for fname in specs:
            with open(fname) as infile:
                outfile.write(infile.read())
                
    with open(path + 'allCH-peaklist.txt', 'w') as outfile:
        for fname in peak_list:
            if os.path.exists(f"{fname}-peaklist-ed.txt"):
                fname = f"{fname}-peaklist-ed.txt"
            else:
                fname = f"{fname}-peaklist.txt"
            with open(fname) as infile:
                outfile.write(infile.read())

    with open(path + 'allCH-lineID.txt', 'w') as outfile:
        for fname in line_list:
            with open(fname) as infile:
                outfile.write(infile.read())
                
   
    return
    
def getLines(peaklist, linelist, species, outfile):
    '''
    Run to get a list of all lines and properties of particular species (i.e. HI) in your list
    Note will simply pick lines identified as particular species or have as next best guess
    May not be 100% accurate and should be checked thuroughly
    
    Parameters
    --------------
    peaklist : str
        path to peaklist file you wish to pull from
    linelist : str
        path to lineID file you wish to pull from
    species : str
        The line species you want i.e. HI or H2
    outfile : str
        The desired path for the file to be outputted
        will add the extension -{species}_peaklist.txt to the file
    
    
    Returns
    --------------
    .txt file containing the line ID and propoerties of the associated peak
    
    '''
    
    # read in the peaklist file
    cols = np.loadtxt(peaklist, comments="%", skiprows=3, usecols=range(16), unpack=True)
    #Want the wavelength easily accessible for testing
    wave = cols[1]
    peak_inten = cols[2]
    gaussk = cols[3]
    gaussFWHM = cols[4]
    gaussx = cols[5]
    Inten = cols[6]
    Flux = cols[7]
    Ap = cols[8]
    omega = cols[9]
    egaussk = cols[10]
    egaussS = cols[11]
    egaussx = cols[12]
    eInten = cols[13]
    eflux = cols[14]
    flag = cols[15]
    
    # read in linelist file
    line = readLineID(linelist)
    best_labels = [b["best"]["label"] if b.get("best") else "" for b in line["blocks"]]
    next_labels = [b["next"]["label"] if b.get("next") else "" for b in line["blocks"]]
    

    if len(wave) != len(best_labels):
        print(f"⚠️ Warning: mismatch between wave ({len(wave)}) and best_labels ({len(best_labels)}) lengths")
    
    with open(f'{outfile}-{species}_peaklist.txt', 'w') as fp:
            fp.write("%% ID wave(um) peak(MJy/sr) GaussK(MJy/sr) GaussFWHM(um) GaussX(um) Intensity(W/m2/sr) Flux(W/m2) " 
                     "ApRadius(arcsec) SolidAngle(omega) eGaussK eGaussS eGaussX eIntensity eFlux flag\n")
            for i in range(len(wave)):
                if best_labels[i].startswith(species):
                    label = best_labels[i]
                    fp.write(f"{label} {wave[i]:8.4f} {peak_inten[i]:12.4f} {gaussk[i]:12.4f} "
                         f"{gaussFWHM[i]:8.4f} {gaussx[i]:8.4f} {Inten[i]:8.4g} "
                         f"{Flux[i]:8.4g} {Ap[i]:8.4g} {omega[i]:8.4g} "
                         f"{egaussk[i]:8.4g} {egaussS[i]:8.4g} {egaussx[i]:8.4g} "
                         f"{eInten[i]:8.4g} {eflux[i]:8.4g} {flag[i]} \n")
                elif next_labels[i].startswith(species):
                    label = next_labels[i]
                    fp.write(f"{label} {wave[i]:8.4f} {peak_inten[i]:12.4f} {gaussk[i]:12.4f} "
                         f"{gaussFWHM[i]:8.4f} {gaussx[i]:8.4f} {Inten[i]:8.4g} "
                         f"{Flux[i]:8.4g} {Ap[i]:8.4g} {omega[i]:8.4g} "
                         f"{egaussk[i]:8.4g} {egaussS[i]:8.4g} {egaussx[i]:8.4g} "
                         f"{eInten[i]:8.4g} {eflux[i]:8.4g} {flag[i]} \n")
    
    
    return    
    
    
    
     
    