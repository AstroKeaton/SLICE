# SLICE 

SLICE (Spectral Line Identification in Cube Environments) is a Python package for extracting spectra and identifying lines from Fits cubes.
It is developed with JWST data in mind but should be applicable to all IR data. 


----------------------------------------------------

## Primary functions

- **catPeaks** - Main code for extracting spectrum and cataloging peaks.
- **cullLines** - Code to cut specified peaks found from catPeaks.
- **keepLines** - Code to keep lines from catPeaks found around specified wavelengths.
- **lineID** - Parses through line lists to identify lines i.e. peaks cataloged by catPeaks.
- **findLines** - Detect and (optionally) identify lines in a specified wavelength range.
- **findSpecies** - Searches ID'd lines for all lines of a given species (eg. CO, H2).
- **readLineID** - Reads and parses line identification results into a Python dictionary.
- **plotID** - Quick plotting of spectra with identified lines marked and labeled.


## How to Install 

### Required packages

SLICE requires:
- numpy
- scipy
- matplotlib
- astropy
- pandas

### Developer install (editable mode)
If you plan to modify the source code, install in editable mode:

```bash
git clone https://github.com/yourusername/SLICE.git
cd SLICE
pip install -e .
```

## Example workflow

### Step 1: Importing
Importing can be done multiple ways depending on user preference

```python
import SLICE
```
or
```python
from SLICE import catPeaks, cullLines, lineID, plotID
```
The former imports all of the primary functions to be called via SLICE.{insert function}.
The latter calls the specific functions the user wants to use

### Step 2: Extract sepctrum and catalog peaks
To extract the spectrum from a data cube we either run catPeaks
or findLines (for user defined wavelength range rather than full range of cube)

```python
catPeaks(fname="Example_cube.fits" , outpath='path_gohere", 
         pixco = ['HH:MM:SS','HH:MM:SS',rad_arcsec], snr=3)

```
This will return multiple .txt files including the spectrum, the continuum, the model of the spectrum,
and a list of identied peaks with their properties. A plot of the peaks marked on the spectrum is generated as well

### Step 3: Cutting lines
Say there's a particular peak that the code ID'd that you think is likely not a true line. 
It can be cut using cullLines by calling the .peaklist file from the last outpath (no extension needed).
Simply provide the same path as the set outpath and then give it a list of lines to cut.
The list will interpret the line as the same as the line number i.e. (line 1 = # 1)

```python
cullLines('path_gohere',[1,6,11]) 
```

### Step 4: Identify cataloged lines
When satisfied with peak list can run lineID to parse catalogs and identify the most likely line
for each peak. The user must define the VLSR which will be treated as redshift instead at <20.
This will generate a lineID.txt file which lists possible lines for each peak. 

```python
lineID('path_gohere', vlsr=200)
```
### Step 5: Plotting ID lines
To plot the spectrum with marked line ID's run plotID

```python
plotID('path_gohere-lineID.txt', specname='path_gohere_spec.txt')
```


## Included Line Lists
SLICE ships with multiple catalogs stored in `line_lists/`:
- Atomic/Ionic fine-structure lines
- Hydrogen and Helium transitions
- CO rotational lines
- ISO line list
- NIST atomic database
- PDR4all lines



## Authors

Base functions and initial code written in Matlab by: A. Bolatto

Adapted into Python and additional commands by: K. Donaghue


## Contributors
This code is open source and open to contribution

### Current Contributors



## Liscense

This project is licensed under the terms of the [MIT License].  
You are free to use, modify, and distribute this software with attribution.