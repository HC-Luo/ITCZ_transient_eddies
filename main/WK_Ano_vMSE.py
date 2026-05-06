#monospectrum analysis of v'MSE', frequency vs wavenumber
import numpy as np 
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import myfun as wf
import math, gc, logging,os,copy
import pandas as pd
from scipy.stats import chi2

def read_data(fnames, BOUNDARY,temporal_resolution,spatial_resolution):
    ds = xr.open_mfdataset(fnames,concat_dim = 'time', combine='nested',decode_times=False)
    #slicing and regridding
    if (len(ds.latitude) > 1) & (ds.latitude[0] > ds.latitude[1]):
        ds = ds.reindex(latitude=ds.latitude[::-1])
    builtin_spatial_resolution = (ds.longitude[1]-ds.longitude[0]).values

    TIME = ds['time']
    builtin_temporal_resolution = (TIME[1]-TIME[0]).values

    ds = ds.sel(time = slice(TIME[0],TIME[-1],int(temporal_resolution/builtin_temporal_resolution)),
                latitude = slice(min(BOUNDARY[0],BOUNDARY[1]),max(BOUNDARY[0],BOUNDARY[1]),int(spatial_resolution/builtin_spatial_resolution)), 
                longitude = slice(min(BOUNDARY[2],BOUNDARY[3]),max(BOUNDARY[2],BOUNDARY[3]),int(spatial_resolution/builtin_spatial_resolution)))
    logging.info("Finishing reading data")
    return ds

def smooth_1_2_1(array, dim, niter):
    arr = array.copy()
    for _ in range(niter):
        shifted_minus = arr.shift({dim: -1}, fill_value=np.nan)
        shifted_plus = arr.shift({dim: 1}, fill_value=np.nan)
        arr = (shifted_minus + 2 * arr + shifted_plus) / 4
    return arr

def wf_analysis(x, **kwargs):
    """Return normalized spectra of x using standard processing parameters."""
    # Get the "raw" spectral power
    # OPTIONAL kwargs: 
    # segsize, noverlap, spd, latitude_bounds (tuple: (south, north)), dosymmetries, rmvLowFrq

    z2 = wf.spacetime_power(x, **kwargs)
    z2avg = 0.5*z2.sum(dim='component')
    z2.loc[{'frequency':0}] = np.nan # get rid of spurious power at \nu = 0

    # separate components
    z2_sym = z2[0,...]#wf.smooth_wavefreq(z2[0,...], kern=wf.simple_smooth_kernel(), nsmooth=1, freq_name='frequency')
    z2_asy = z2[1,...]#wf.smooth_wavefreq(z2[1,...], kern=wf.simple_smooth_kernel(), nsmooth=1, freq_name='frequency')


    background = copy.deepcopy(z2avg)
    background = smooth_1_2_1(background, 'frequency', 10)
    background = smooth_1_2_1(background, 'wavenumber', 40)

    nspec_sym = 1-background/z2_sym
    nspec_asy = 1-background/z2_asy
    # Compute the maximum value in the entire 2D array
    positive_freq_data = background.sel(frequency=background['frequency'] > 0)
    max_val = positive_freq_data.max()
    
    # Create a boolean mask of where the DataArray equals the maximum
    mask = positive_freq_data == max_val
    
    # Extract the coordinates where the mask is True
    # Because it could happen in multiple locations, we take the first occurrence:
    max_idx = mask.where(mask, drop=True).stack(points=("wavenumber", "frequency"))
    
    # Get the first matching coordinate values
    freq_at_max = max_idx["frequency"].values[0]
    wavenum_at_max = max_idx["wavenumber"].values[0]
    
    print(f"Maximum value is {max_val.item()} at frequency={freq_at_max}, wavenumber={wavenum_at_max}")

    return nspec_sym, nspec_asy


if __name__ == "__main__":
    fpath1 = "/glade/derecho/scratch/hcluo/"
    fpath2 = "/glade/campaign/univ/uccn0006/q_col/"

    years = np.arange(1997,2024)
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    MONTHS = ["%02d" % x for x in np.arange(1,13)]
    days = ["%02d" % x for x in np.arange(1,32)]
    a = 6371.0*1e3
    BOUNDARY = [-20,20,0,360]
    spatial_resolution = 0.5 #degree
    temporal_resolution = 12 #hrs, can only be 1, 2, 3, 4, 6, 12
    fnames2 = []
    fnames4 = []
    for i in years:
        fname2 = "vMSE_col_transient/vMSE_col_transient_"+f"{i}.nc"
        fnames2.append(fpath1+fname2)

    ds2 = read_data(fnames2,BOUNDARY,temporal_resolution,spatial_resolution)
    ds2['time'] = ds2['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    data_selected = xr.decode_cf(ds2,decode_times = True)['vMSE_transient_col']
    # Create a mask: True where we want to keep data, False elsewhere
    
    #data_selected[...,260:] = 0
    #data_selected[...,0:120] = 0
    
    #data_selected[...,178:] = 0
    #data_selected[...,:177] = 0
    #data_selected[...,357:] = 0
    #data_selected[...,0:50] = 0
    #data_selected[...,0:35,...] = 0
    Data_selected = data_selected.sel(time = data_selected.time.dt.month.isin(months))
    #Data[:,0:15,:] = -1*Data[:,0:15,:]

    logging.info(f"data is {Data_selected}")
    # Options ... right now these only go into wk.spacetime_power()
    #
    latBound = (BOUNDARY[0],BOUNDARY[1])  # latitude bounds for analysis
    spd      = int(24/temporal_resolution)    # SAMPLES PER DAY
    nDayWin  = 96   # Wheeler-Kiladis [WK] temporal window length (days)
    nDayOverlap = 60  # time (days) between temporal windows [segments]
                        # positive means there will be overlapping temporal segments

    opt      = {'segsize': nDayWin, 
                'noverlap': nDayOverlap, 
                'spd': spd, 
                'dosymmetries': True, 
                'rmvLowFrq':True}
    # in this example, the smoothing & normalization will happen and use defaults
    symComponent, asymComponent = wf_analysis(Data_selected, **opt)
    #
    # Plots ... sort of matching NCL, but not worrying much about customizing.
    #

    # significance test
    n = 2*len(Data_selected.latitude)*len(Data_selected.time)/spd/nDayWin
    chi_square = chi2.ppf(0.95, n, loc=0, scale=1)
    rc = 1-(n-1)/chi_square
    symComponent = symComponent.where(symComponent > rc, np.nan)
    asymComponent = asymComponent.where(asymComponent > rc, np.nan)

    cmap = wf.colormap(color="WhiRed")

    outPlotName = "/glade/work/hcluo/v5/Ano_vMSE_symmetric_-20_20.pdf"
    wf.plot_normalized_symmetric_spectrum(symComponent, cmap, 
                                          levels = np.linspace(0.02, 0.2, 10),
                                          ofil = outPlotName)
    
    outPlotName = "/glade/work/hcluo/v5/Ano_vMSE_asymmetric_-20_20.pdf"
    wf.plot_normalized_asymmetric_spectrum(asymComponent, cmap, 
                                           levels = np.linspace(0.02, 0.2, 10),
                                           ofil = outPlotName)
    