'''
Code originally written in MATLAB by: A. Bolatto

Adapted and modified for use in Python by: K. Donaghue
Last modified: 9/1/2025

This code is best applied to JWST or similar IR data

Primary functions:
-catPeaks
    -Take fits cubes and extract spectra of set region
    -Identify peaks in spectrum and fit gaussian to them
    -Record key properties of lines such as intensity and flux
    -Produces among other things a line list to be used for IDs
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
'''


import numpy as np
import re
from typing import List, Dict, Any

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



def gaussmfit(x, y, K, S, X0, tol=1e-7, max_iter=200, lambda_init=1e-3, verbose=False):
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
    from numpy.linalg import solve, inv
    
    # Convert inputs
    x, y = np.asarray(x, float), np.asarray(y, float)
    K, S, X0 = np.asarray(K, float), np.asarray(S, float), np.asarray(X0, float)
    b, d = len(K), len(x)

    # Expand to matrices for vectorized math
    x_mat = np.tile(x, (b, 1))
    y_mat = np.tile(y, (b, 1)) / b

    def model(Kv, Sv, X0v):
        """Gaussian model (vectorized for multiple peaks)."""
        return Kv[:, None] * np.exp(-((x_mat - X0v[:, None])**2) / (2 * Sv[:, None]**2))

    # Initial model and chi2
    ymodel = model(K, S, X0)
    delta = y_mat - ymodel
    chi2 = np.sum(delta**2)
    
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
                    [np.sum(dKi * dKj) + L*diagV, np.sum(dKi * dSj), np.sum(dKi * dX0j)],
                    [np.sum(dSi * dKj), np.sum(dSi * dSj) + L*diagV, np.sum(dSi * dX0j)],
                    [np.sum(dX0i * dKj), np.sum(dX0i * dSj), np.sum(dX0i * dX0j) + L*diagV]
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

        if newchi2 < chi2:  # Accept step
            converge_val = (chi2 - newchi2) / newchi2
            if verbose:
                print(f"Iter {it:3d} | chi2={newchi2:.4e} | Δχ²/χ²={converge_val:.2e} | L={L:.2e}")
            if converge_val < tol:
                break  # Converged
            L /= 10
            K, S, X0, ymodel, delta, chi2 = newK, newS, newX0, newymodel, newdelta, newchi2
        else:  # Reject step
            L *= 10
            if L > 1e3:
                err_code = 1  # L exceeded
                break

    # Collapse multi-component fit into final ymodel
    if b > 1:
        ymodel = np.sum(ymodel, axis=0)

    # Reduced chi-square
    dof = max(1, d - 3*b)
    chi2 /= dof

    # Compute covariance matrix and parameter uncertainties
    emat = inv(alpha)
    epar = np.sqrt(chi2 * np.abs(np.diag(emat)))

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
    Fits reader for acquiring necessary data for catPeaks.
    
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
            return {'header': header}

        data = hdu.data.astype(float)  # ensure float for blank handling
        if 'BLANK' in header:
            blankval = header['BLANK']
            data[data == blankval] = bval

        # Scale if needed
        bscale = header.get('BSCALE', 1.0)
        bzero = header.get('BZERO', 0.0)
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
        naxis = header.get('NAXIS', 0)
        axes = []
        for i in range(naxis):
            axis_len = header.get(f'NAXIS{i+1}', 1)
            crval = header.get(f'CRVAL{i+1}', 0.0)
            cdelt = header.get(f'CDELT{i+1}', 1.0)
            crpix = header.get(f'CRPIX{i+1}', 1.0)
            axis = (np.arange(axis_len) + 1 - crpix) * cdelt
            if not noabs:
                axis += crval
            axes.append(axis)

        result = {
            'header': header,
            'data': data,
            'x': axes,
            'bunit': header.get('BUNIT', ''),
            'naxis': naxis,
            'bitpix': header.get('BITPIX', None),
            'numpt': [header.get(f'NAXIS{i+1}', 0) for i in range(naxis)],
            'crval': [header.get(f'CRVAL{i+1}', 0.0) for i in range(naxis)],
            'crpix': [header.get(f'CRPIX{i+1}', 1.0) for i in range(naxis)],
            'cdelt': [header.get(f'CDELT{i+1}', 1.0) for i in range(naxis)],
            'cunit': [header.get(f'CUNIT{i+1}', '') for i in range(naxis)],
            'ctype': [header.get(f'CTYPE{i+1}', '') for i in range(naxis)],
            'wcs': wcs,
            'coormode': 'ABSCOOR' if not noabs else 'RELCOOR',
        }

        # Optional WCS coordinate arrays
        if wcs is not None and naxis >= 2:
            ny, nx = data.shape[-2], data.shape[-1]
            yy, xx = np.mgrid[:ny, :nx]
            try:
                ra, dec = wcs.celestial.pixel_to_world(xx, yy).ra.deg, \
                          wcs.celestial.pixel_to_world(xx, yy).dec.deg
                result['wcs1'] = ra
                result['wcs2'] = dec
            except Exception as e:
                print(f"WCS conversion failed: {e}")

        return result
    
    



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


def extract_aperture(r, pixco, useapprox=False, silent=False):
    """
    Extract a spectrum using an aperture on the FITS data cube.
    
    function is used for catPeaks
    
    Parameters:
    - r: dictionary containing FITS data and WCS (from rfits)
    - pixco: pixel coordinates, list of coordinates, or circular region spec
    - useapprox: if True, uses crude radial cutoff instead of accurate fractional aperture
    - silent: if True, suppresses warnings

    Returns:
    - s: extracted 1D spectrum
    - region: string identifier for aperture type
    - mask: 2D mask of used pixels
    - flag: status flag (0 = OK, -1 = outside cube, -2 = too many NaNs)
    """
    import numpy as np
    from numpy import cos, deg2rad
    from astropy.wcs.utils import skycoord_to_pixel
    from astropy.coordinates import SkyCoord
    import warnings
    
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

    pixco = np.array(pixco)

    # Handle 'pixel' case
    if pixco.ndim == 1 and pixco.shape[0] == 2:
        region = 'pixel'
        s = data[pixco[0], pixco[1], :]
        mask[pixco[0], pixco[1]] = 1

    # Handle 'list' case
    elif len(pixco.shape)>1:
        if pixco.shape[0] == 2 and pixco.shape[1] > 1:
            region = 'list'
            s = np.nanmedian(data[pixco[0], pixco[1], :], axis=(0,))
            mask[pixco[0], pixco[1]] = 1

    # Handle 'all' case
    elif pixco.size == 1:
        region = 'all'
        s = np.nanmedian(data, axis=(0, 1))
        mask[:] = 1

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
            raise ValueError("Format should be ['00:00:00.0','00:00:00.0', 0.0] or [0.0, 0.0, 0.0]")

        wcs1 = r["wcs1"]
        wcs2 = r["wcs2"]

        xx = (wcs1 - rac) * np.cos(np.deg2rad(r["crval"][1])) * 3600
        yy = (wcs2 - dec) * 3600

        if useapprox:
            dist = np.sqrt(xx**2 + yy**2)
            ix = np.where(dist <= rad)
        else:
            
            weight = aperture(xx, yy, 0, 0, r["cdelt"][1] * 3600, rad)
            sw = weight / np.nansum(weight)

        s = np.zeros(shape[2])
        ck = np.zeros(shape[2])
        for k in range(shape[2]):
            if useapprox:
                p = data[:, :, k]
                s[k] = np.nanmean(p[ix])
            else:
                p = data[:, :, k] * sw
                pck = np.ones_like(p)
                pck[np.isnan(p)] = np.nan
                pck *= sw
                ck[k] = np.nansum(pck)
                s[k] = np.nansum(p)
                if s[k] == 0:
                    s[k] = np.nan
                if ck[k] < 0.25:
                    s[k] = np.nan
                else:
                    s[k] = s[k] / ck[k]

        if np.all(np.isnan(s)):
            flag = -1
            if not silent:
                warnings.warn("Aperture is empty (probably outside the cube)")
            return s, region, mask, flag
        if np.sum(np.isnan(s)) / len(s) > 0.05:
            flag = -2
            if not silent:
                warnings.warn("A lot of NaNs in the spectrum (close to an edge)")
            return s, region, mask, flag

        if useapprox:
            mask[ix] = 1
        else:
            mask = weight

    else:
        raise NotImplementedError("Averaging dimensions not implemented")

    return s, region, mask, flag


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

    dix = np.diff(ix)
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




def fitLines(wv, spec, peaks):
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
    base_interp = interp1d(wv[include], spec[include], kind='linear', fill_value="extrapolate")
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

        Ko, So, X0o, chi2, yz, err, epar = gaussmfit(wvreg, spreg, K, S, X0, 1e-7)

        ix_self = np.where(contrib == i)[0][0]
        q['K'][i] = Ko[ix_self]
        q['eK'][i] = epar[ix_self * 3]
        q['S'][i] = So[ix_self]
        q['eS'][i] = epar[ix_self * 3 + 1]
        q['X0'][i] = X0o[ix_self]
        q['eX0'][i] = epar[ix_self * 3 + 2]

        nu = c / (1e-6 * q['X0'][i])
        dnu = nu * q['S'][i] / q['X0'][i]
        ednu = np.sqrt(
            (q['S'][i]**2 / q['X0'][i]**4) * q['eS'][i]**2 +
            (4 * q['S'][i]**2 / q['X0'][i]**6) * q['eX0'][i]**2
        )

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

    return q, ymodel, smo



def catPeaks(fname, outname, pixco=None, snr=5, thr=2, nchunk=1):
    '''
    meant to plot and mark peaks from a spectrum.
    creates catalog of significant peaks for a spectrum
    
    Parameters:
    ---------------
    fname : str 
        Name of input file. Typically fits file.
    outname : str 
        Name of output file. formats as .txt file.
    pixco : array or list like 
        If "pixco" is an array of 1 row and 2 columns i.e. [X,Y] it will assume the first 
        component is a row number and the second a column number for the pixel in
        the data, and extract that spectrum to analyze. If it has two rows i.e.
        [X1 , X2, X3], [Y1, Y2, Y3], it will assume that the first row is the list of 
        row numbers and the second thelist of column numbers of the area to average over. 
        If it is a list with two strings and a float (i.e. [RA, Dec, rad]) it will extract a spectrum at 
        that RA,DEC (colon-separated sexagesimal) in an aperture of that radius in arcsec.
    snr : float 
        Minimum signal to noise for significant detection. Defaults to 5.
    thr : float 
        minimum separation in channels between significant areas in spectrum. Defaults to 2.
    nchunk : int 
        The number of chunks to cut the spectrum into, it defaults to 1. 
        If it is a vector the first and second component will be the start-end
        wavelengths (um) of the analysis.
    
    -------
    Returns: .txt file containing peak information
             .txt file containing the extracted spectra
             .txt file containing the continuum of region
             .tex file containing the model of the spectra
             .png plot of spectra with peaks marked and labeled
    '''
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Handle input
    if isinstance(fname, str):
        #assumes fits cube has additional extension where data is housed
        r = rfits(fname, 'xten1')  
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

    s, region, mask, flag = extract_aperture(r, pixco)
    if flag != 0:
        return {'flag': flag}

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
    q_result = {
        'S': [], 'K': [], 'X0': [], 'area': [],
        'eK': [], 'eS': [], 'eX0': [], 'earea': [], 'flag': []
    }

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
        q, ymo, cont = fitLines(xx, yy, p)

        np_detected += len(p)
        pp.extend(p)

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
        plt.close()

        # Print peaks
        for j, (loc, amp) in enumerate(p):
            print(f"{onp + j:3d}   {loc:.4f} um {amp:8.4f} {r['bunit']}")
        onp = np_detected + 1

        # Append fit info
        for k in range(len(q['S'])):
            for key in q_result:
                q_result[key].append(q[key][k])

    # Write output files
    s2f = np.sqrt(np.log(256))
    with open(f'{outname}-peaklist.txt', 'w') as fp:
        fp.write(f"# Cube: {fname}\n")
        fp.write(f"# Region: {region}\n")
        if region == 'pixel':
            fp.write(f"pixel {pixco[0]}, {pixco[1]}\n")
        elif region == 'list':
            fp.write("list " + " ".join(map(str, pixco[0])) + "\n")
            fp.write("#              " + " ".join(map(str, pixco[1])) + "\n")
        elif region == 'circle' and isinstance(pixco, list):
            fp.write(f"circle {pixco[0]},{pixco[1]},{pixco[2]}\n")
        else:
            fp.write("# undefined\n")

        fp.write(f"# num wave(um) peak({r["bunit"]}) GaussK({r["bunit"]}) GaussFWHM(um) GaussX(um) Flux(W/m2/sr) eGaussK eGaussS eGaussX eFlux flag\n")
        for i in range(np_detected):
            S_abs = abs(q_result['S'][i])
            flux = q_result['area'][i]
            eS = q_result['eS'][i] * s2f
            fp.write(f"{i+1:3d} {pp[i][0]:8.4f} {pp[i][1]:12.4f} {q_result['K'][i]:12.4f} "
                     f"{S_abs * s2f:8.4f} {q_result['X0'][i]:8.4f} {flux:8.4g} "
                     f"{q_result['eK'][i]:8.4g} {eS:8.4g} {q_result['eX0'][i]:8.4g} "
                     f"{q_result['earea'][i]:8.4g} {q_result['flag'][i]:2d}\n")

    # Spectrum output
    np.savetxt(f'{outname}_spec.txt', np.column_stack((xx, yy)), header=f'{fname}')
    np.savetxt(f'{outname}_cont.txt', np.column_stack((xx, cont)), header=f'{fname}')
    np.savetxt(f'{outname}_model.txt', np.column_stack((xx, ymo)), header=f'{fname}')

    return q_result

###################################################
#########Commands to manipulate line list##########
def recomputeLines(file):
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
    with open(fname, "r") as fp:
        for _ in range(nh):
            ln = fp.readline().strip()
        if "Gauss" in ln:
            cols = np.loadtxt(fp, comments="%", ndmin=2)
            # Expecting 12 columns
            C = [cols[:, i] for i in range(cols.shape[1])]
        else:
            cols = np.loadtxt(fp, comments="%", ndmin=2)
            # Expecting 3 columns
            C = [cols[:, i] for i in range(cols.shape[1])]

    # --- Load spectrum ---
    u = np.loadtxt(f"{file}_spec.txt")

    # --- Fit lines ---
    q, ymo, cont = fitLines(u[:, 0], u[:, 1], np.column_stack([C[1], C[2]]))

    # --- Derived quantities ---
    s2f = np.sqrt(np.log(256))
    c = 3e8
    nu = c / (1e-6 * q["X0"])
    dnu = nu * q["S"] / q["X0"]
    ednu = np.sqrt((q["S"]**2 / q["X0"]**4) * (q["eS"]**2) +
               (4 * q["S"]**2 / q["X0"]**6) * (q["eX0"]**2))
    area = 1e-20 * np.sqrt(2 * np.pi) * q["K"] * dnu
    earea = 1e-20 * np.sqrt(2 * np.pi) * np.sqrt((dnu**2) * (q["eK"]**2) +
                                             (q["K"]**2) * (ednu**2))


    
    # Combine output table
    K = np.column_stack([
        C[0].astype(int),     # ID
        C[1],                 # col2
        C[2],                 # col3
        q['K'],
        q['S'] * s2f,
        q['X0'],
        area,
        q['eK'],
        q['eS'] * s2f,
        q['eX0'],
        earea,
        q['flag'].astype(int)
    ])

    # --- Rewrite peaklist file ---
    with open(fname, "w") as fpw:
        for h in hl:
            fpw.write(f"{h}\n")
        for row in K:
            fpw.write((
                f"{int(row[0]):3d} {row[1]:8.4f} {row[2]:12.4f} {row[3]:12.4f} "
                f"{row[4]:8.4f} {row[5]:8.4f} {row[6]:8.4g} {row[7]:8.4g} "
                f"{row[8]:8.4g} {row[9]:8.4g} {row[10]:8.4g} {int(row[11]):2d}\n"
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



def keepLines(file, wave_list, tol=0.05):
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
        header = [line for line in fp if line.startswith("%")]
    if "Gauss" in ln:
        ncols = 12
    else:
        ncols = 3

    # --- Step 2: load numeric data
    cols = np.loadtxt(
        fname,
        comments="%",
        skiprows=3,
        usecols=range(ncols),
        
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
                    f"{row[4]:8.4f} {row[5]:8.4f} {row[6]:8.4g} {row[7]:8.4g} "
                    f"{row[8]:8.4g} {row[9]:8.4g} {row[10]:8.4g} {int(row[11]):2d}\n"
                )

    
    print(f'Written {outname}')
    recomputeLines(file)
    return




def cullLines(file, remove_list):
    """
    Remove specific line indices from a peaklist and force creation of
    an '-ed' peaklist file.

    Parameters
    ----------
    file : str
        Base filename (without -peaklist suffix).
    remove_list : array_like
        List of line indices to remove.
    
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

    if "Gauss" in ln:
        ncols = 12
    else:
        ncols = 3

    # --- Step 2: load numeric data
    cols = np.loadtxt(
        fname,
        comments="%",
        skiprows=3,
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
    recomputeLines(file)
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




#####################################################################################
#####LineID code ####################################################################
def lineID(linefile, vlsr, R=250, outname=None, cat_path=None, skiprows=3):
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
    skiprows : int
        number of rows to skip in linefile, includes comments
        
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
                cols = np.loadtxt(fname, skiprows=skiprows, usecols=range(12), unpack=True)
            else:
                cols = np.loadtxt(fname, skiprows=skiprows, usecols=range(3), unpack=True)
        lineno, wv = cols[0].astype(int), cols[1]
    elif isinstance(linefile, (list, np.ndarray)):
        wv = np.array(linefile)
        lineno = np.arange(1, len(wv) + 1)
        linefile = "default"
        if outname is None:
            outname = f"{linefile}-lineID.txt"
    else:
        raise ValueError("First parameter type not supported.")

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

    cat = []
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
        print(f"#{lnum:3d} {w:8.4f}+/-{err:0.4f} um (rest={wvr[i]:8.4f} um):\n       ")

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
                    print(f"| {tn[idx]} {velshift:+4.0f} | ", end="")
                else:
                    fp.write(f"{tn[idx]} {velshift:+4.0f} | ")
                    print(f"{tn[idx]} {velshift:+4.0f} | ", end="")
            fp.write("\n\n")
            print()

            fp.write(f"       BEST: DLambda={dd[ix[0]]:0.4f} um, Catalog={catnames[cn[ix[0]]]}, Restwv={rv[ix[0]]:0.4f}\n       {tn[ix[0]]}\n")
            print(f"       BEST: DLambda={dd[ix[0]]:0.4f} um, Catalog={catnames[cn[ix[0]]]}, Restwv={rv[ix[0]]:0.4f}\n       {tn[ix[0]]}")
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
                    print(f"       NEXT: DLambda={dd[ix[k]]:0.4f} um, Catalog={catnames[cn[ix[k]]]}, Restwv={rv[ix[k]]:0.4f}\n       {tn[ix[k]]}")
        else:
            fp.write(f"       NOMATCH within +/-{w/R:0.4f} um\n")
            print(f"       NOMATCH within +/-{w/R:0.4f} um")
            nm += 1

    fp.close()
    print(f"Attempted IDs for {len(wv)} lines, found no matches for {nm} lines")
    return

######## Additional commands ####################

def findLines(fname, outname, wmin, wmax, pixco=None, snr=5, thr=2, ID=False, vlsr=None, catdir=None):
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
    snr, thr : float
        Peak finding parameters.(see catPeaks)
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
    s, region, mask, flag = extract_aperture(r, pixco)
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

    q_result, ymo, cont = fitLines(xx, yy, p)

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
        fp.write("%% num wave(um) peak(MJy/sr) GaussK GaussFWHM(um) GaussX(um) "
                 "Flux(W/m2/sr) eGaussK eGaussS eGaussX eFlux flag\n")
        for i in range(len(p)):
            S_abs = abs(q_result['S'][i])
            flux = q_result['area'][i]
            eS = q_result['eS'][i] * s2f
            fp.write(f"{i+1:3d} {p[i][0]:8.4f} {p[i][1]:12.4f} {q_result['K'][i]:12.4f} "
                     f"{S_abs * s2f:8.4f} {q_result['X0'][i]:8.4f} {flux:8.4g} "
                     f"{q_result['eK'][i]:8.4g} {eS:8.4g} {q_result['eX0'][i]:8.4g} "
                     f"{q_result['earea'][i]:8.4g} {q_result['flag'][i]:2d}\n")

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
            cols = np.loadtxt(fname, comments="%", skiprows=3, usecols=range(12), unpack=True)
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
            plt.ylabel("Flux (MJy/sr)")

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
    
    hdr = np.loadtxt(specname,dtype=str,delimiter=' ',comments='%',skiprows=6,max_rows=1)
    bunit = hdr[-1].replace('}','').replace("'","")
    basedata = np.loadtxt(specname)#,skiprows=skiprows,usecols=(1,2))
    x, y = basedata[:, 0], basedata[:, 1]
    top = max(y)
    
    res = readLineID(linename)
    best_labels = [b["best"]["label"] for b in res["blocks"] if b["best"]]
    obwave = [k['obs_wv_um'] for k in res['blocks'] if k['best']]
    
    plt.figure(figsize=(9,8))
    plt.plot(x, y, color='k', lw=1, zorder=10)
    # plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Wavelength ($\\mu$m)')
    plt.ylabel('Flux ('+bunit+')')
    plt.minorticks_on()
    
    for j in range(len(obwave)):
        plt.axvline(obwave[j], color='r', ls='-', alpha=0.2, zorder=5)
        plt.text(obwave[j]-0.05, top*0.5, best_labels[j], rotation=90,
                 ha='center', va='bottom', fontsize=8, color='red')
        
    if savefig is True:
        if outname is None:
            raise ValueError('Must define a path for figure to save to')
        plt.savefig(outname, bbox_inches='tight')
    else:
        plt.show()
    
    
    return


    
    
    
    
    
    
     
    