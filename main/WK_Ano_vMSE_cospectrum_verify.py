#plot the ratio of TD signal over the full band
import numpy as np 
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import myfun as wf
import math, gc, logging,os,copy
import pandas as pd
from scipy.stats import chi2
from scipy.signal import convolve2d, detrend

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

def spacetime_power(data1,data2, segsize, noverlap, spd, dosymmetries=True, rmvLowFrq=True):
    segsize = spd*segsize
    noverlap = spd*noverlap
    assert segsize-noverlap > 0, f"Error, inconsistent specification of segsize and noverlap results in stride of {segsize-noverlap}, but must be > 0."
    Ks = []
    for data in [data1,data2]:
        slat = data['latitude'].min().item()
        nlat = data['latitude'].max().item()
        
        # "Remove dominant signals"
        # "detrend" the data, including removing the mean (uses scipy.signal.detrend):
        data = data.transpose('time', 'latitude', 'longitude')
        xdetr = detrend(data, axis=0, type='linear')
        xdetr = xr.DataArray(xdetr, dims=data.dims, coords=data.coords)
        
        # filter low-frequencies    
        if rmvLowFrq:
            data = wf.rmvAnnualCycle(xdetr, spd, 1/(segsize/spd))
            logging.info("Annual cycle is removed")
    
        dimsizes = data.sizes  # dict
        lon_size = dimsizes['longitude']
        lat_size = dimsizes['latitude']
        lat_dim = data.dims.index('latitude')
        if dosymmetries:
            data = wf.decompose2SymAsym(data)
            logging.info("Decomposited into symetric and asymetric")
        # testing: pass -- Gets the same result as NCL.
    
        # 2. Windowing with the xarray "rolling" operation, and then limit overlap with `construct` to produce a new dataArray.
        # WK99 recommend "2-month" overlap
        x_win = data.rolling(time=segsize, min_periods=segsize).construct("segments")  # WK99 use 96-day window
        x_win = x_win.isel(time=slice(segsize-1,None,segsize-noverlap))  
    
        logging.debug(f"[spacetime_power] x_win shape is {x_win.shape}")
        # Additional detrend for each segment: means??????????????????????????????????
        if  np.logical_not(np.any(np.isnan(x_win))):
            logging.info("No missing, so use simplest segment detrend.")
            x_win_detr = detrend(x_win, axis=-1, type='linear') #<-- missing data makes this not work
            x_win = xr.DataArray(x_win_detr, dims=x_win.dims, coords=x_win.coords)
        else:
            logging.warning("EXTREME WARNING -- This method to detrend with missing values present does not quite work, probably need to do interpolation instead.")
            logging.warning("There are missing data in x_win, so have to try to detrend around them.")
            x_win_cp = x_win.copy()
            logging.info(f"[spacetime_power] x_win_cp windowed data has shape {x_win_cp.shape} \n \t It is a numpy array, copied from x_win which has dims: {x_win.sizes} \n \t ** about to detrend this in the rightmost dimension.")
            x_win_cp[np.logical_not(np.isnan(x_win_cp))] = detrend(x_win_cp[np.logical_not(np.isnan(x_win_cp))])
            x_win = xr.DataArray(x_win_cp, dims=x_win.dims, coords=x_win.coords)
        
        # 3. Taper in time to make the signal periodic, as required for FFT.
        taper = wf.split_hann_taper(segsize, 0.1)  
        x_wintap = x_win*taper 
        
        # Do the transform using 2D FFT
        #
        # z = np.fft.fft2(x_wintap, axes=(2,3)) / (lon_size * segsize)
    
        # Or do the transform with 2 steps
        z = np.fft.fft(x_wintap, axis=2) / lon_size  # note that np.fft.fft() produces same answers as NCL cfftf
        z = np.fft.fft(z, axis=3) / segsize 
        z = xr.DataArray(z, dims=("time","latitude","wavenumber","frequency"), 
                         coords={"time":x_wintap["time"], 
                                 "latitude":x_wintap["latitude"],
                                 "wavenumber":np.fft.fftfreq(lon_size, 1/lon_size),
                                 "frequency":np.fft.fftfreq(segsize, 1/spd)})
        
        K = wf.resolveWavesHayashi(z, segsize//spd, spd)
        Ks.append(K)
        #Ks.append(z)
    
    z_pee = (Ks[0].real)*(Ks[1].real)+(Ks[0].imag)*(Ks[1].imag)
    # z_pee is spectral power already. 
    # z_pee is a DataArray w/ coordinate vars for wavenumber & frequency

    # average over all available segments and sum over latitude
    # OUTPUT DEPENDS ON SYMMETRIES
    if dosymmetries:
        # multipy by 2 b/c we only used one hemisphere
        z_symmetric = 2.0 * z_pee.isel(latitude=z_pee.latitude<0).mean(dim='time').squeeze()
        z_symmetric.name = "power"
        z_antisymmetric = 2.0 * z_pee.isel(latitude=z_pee.latitude>0).mean(dim='time').squeeze()
        z_antisymmetric.name = "power"
        z_final = xr.concat([z_symmetric, z_antisymmetric], "component")
        z_final = z_final.assign_coords({"component":["symmetric","antisymmetric"]})
    else:
        #latitude = z_pee['latitude']
        #lat_inds = np.argwhere(((latitude <= nlat)&(latitude >= slat))).squeeze()
        z_final = z_pee.mean(dim='time').squeeze()
    return z_final

def wf_analysis(x1,x2, **kwargs):
    """Return normalized spectra of x using standard processing parameters."""
    # Get the "raw" spectral power
    # OPTIONAL kwargs: 
    # segsize, noverlap, spd, latitude_bounds (tuple: (south, north)), dosymmetries, rmvLowFrq
    z2 = spacetime_power(x1,x2, **kwargs)
    
    return z2
    #z2avg = 0.5*z2.sum(dim='component')
    #z2.loc[{'frequency':0}] = np.nan # get rid of spurious power at \nu = 0
#
    ## separate components
    #z2_sym = z2[0,...]#wf.smooth_wavefreq(z2[0,...], kern=wf.simple_smooth_kernel(), nsmooth=1, freq_name='frequency')
    #z2_asy = z2[1,...]#wf.smooth_wavefreq(z2[1,...], kern=wf.simple_smooth_kernel(), nsmooth=1, freq_name='frequency')
#
#
    #background = copy.deepcopy(z2avg)
    #background = smooth_1_2_1(background, 'frequency', 10)
    #background = smooth_1_2_1(background, 'wavenumber', 40)
#
    #nspec_sym = 1-background/z2_sym
    #nspec_asy = 1-background/z2_asy
    ## Compute the maximum value in the entire 2D array
    #positive_freq_data = background.sel(frequency=background['frequency'] > 0)
    #max_val = positive_freq_data.max()
    #
    ## Create a boolean mask of where the DataArray equals the maximum
    #mask = positive_freq_data == max_val
    #
    ## Extract the coordinates where the mask is True
    ## Because it could happen in multiple locations, we take the first occurrence:
    #max_idx = mask.where(mask, drop=True).stack(points=("wavenumber", "frequency"))
    #
    ## Get the first matching coordinate values
    #freq_at_max = max_idx["frequency"].values[0]
    #wavenum_at_max = max_idx["wavenumber"].values[0]
    #
    #print(f"Maximum value is {max_val.item()} at frequency={freq_at_max}, wavenumber={wavenum_at_max}")
#
    #return nspec_sym, nspec_asy

def plot_normalized_symmetric_spectrum(s, cmap,levels=None, ofil=None):
    """Basic plot of normalized symmetric power spectrum with shallow water curves."""
    fb = [0, .8]  # frequency bounds for plot
    # get data for dispersion curves:
    #swfreq,swwn = genDispersionCurves()
    # swfreq.shape # -->(6, 3, 50)
    #swf = np.where(swfreq == 1e20, np.nan, swfreq)
    #swk = np.where(swwn == 1e20, np.nan, swwn)

    fig, ax = plt.subplots()
    c = 'black' # COLOR FOR DISPERSION LINES/LABELS
    z = s.transpose().sel(frequency=slice(*fb), wavenumber=slice(-20,20))
    z.loc[{'frequency':0}] = np.nan
    kmesh0, vmesh0 = np.meshgrid(z['wavenumber'], z['frequency'])

    img = ax.contourf(kmesh0, vmesh0, z, 
                      levels=levels, 
                      cmap=cmap,  extend='both')
    
    #for ii in range(3,6):
    #    ax.plot(swk[ii, 0,:], swf[ii,0,:], linestyle='dashed', color=c)
    #    ax.plot(swk[ii, 1,:], swf[ii,1,:], linestyle='dashed', color=c)
    #    ax.plot(swk[ii, 2,:], swf[ii,2,:], linestyle='dashed', color=c)
    ax.axvline(0, linestyle='dashed', color='lightgray')
    ax.set_xlim([-20,20])
    ax.set_ylim(fb)    
    ax.set_title("Cospectra")
    ax.tick_params(direction='in', which='both')
    ax.set_xlabel(r"Wavenumber ($deg^{-1}$)")
    ax.set_ylabel(r"Frequency ($day^{-1}$)")
    fig.colorbar(img)
    if ofil is not None:
        fig.savefig(ofil, bbox_inches='tight')
    
    return

def power_mask(power,wavenumber_low,wavenumber_hi,freq_low,freq_hi):
    # Create 2D grid of frequencies and wavenumbers
    K_LON,FREQ = np.meshgrid(power.wavenumber.values, power.frequency.values, indexing='ij')

    # Build 2D mask for time and longitude
    mask = ((
        (K_LON  >= wavenumber_low ) & ( K_LON<= wavenumber_hi ) &
        ( FREQ>= freq_low) & ( FREQ<= freq_hi)
    ) | (
        ( K_LON<= -wavenumber_low ) & (K_LON  >= -wavenumber_hi ) &
        (FREQ  <= -freq_low) & ( FREQ>= -freq_hi)
    ))

    # Reshape mask to 3D (broadcast over latitude)
    #mask_3d = mask[np.newaxis,:,:]
    for i in range(len(power.latitude)):
        power[i,:] = power[i,:]* mask
    return power

if __name__ == "__main__":
    fpath1 = "/glade/work/hcluo/pro1/v4/data/"
    fpath2 = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/"
    fpath3 = "/glade/work/hcluo/data/MSE_adjust/"
    fpath4 = "/glade/derecho/scratch/hcluo/MSE/"
    years = np.arange(1997,2024)
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    MONTHS = ["%02d" % x for x in np.arange(1,13)]
    days = ["%02d" % x for x in np.arange(1,32)]
 
    BOUNDARY = [-20,20,0,360]
    spatial_resolution = 1.0 #degree
    temporal_resolution = 12 #hrs, can only be 1, 2, 3, 4, 6, 12
    a = 6.371e6  
    #fnames1 = []
    fnames2 = []
    fnames3 = []
    fnames4 = []
    fnames5 = []
    fnames6 = []
    for i in years:
        for j in np.arange(1,13):
            year_month = str(i)+MONTHS[j-1]
            fname1 = "vMSE_col_"+year_month+".nc"
            fname2 = "ano_vMSE_col_"+year_month+".nc"
            fname4 = "adjustment_"+year_month+".nc"
            #fnames1.append(fpath1+fname1)
            fnames2.append(fpath1+fname2)
            fnames4.append(fpath3+fname4)

            for k in days:
                fname6 = "e5.oper.an.pl.128_131_u.ll025uv." \
                +year_month+k+"00_"+year_month+k+"23.nc"
                fname3 = "e5.oper.an.pl.128_132_v.ll025uv." \
                +year_month+k+"00_"+year_month+k+"23.nc"

                fname5 = "MSE_"+year_month+k+"00_"+year_month+k+"18.nc"
                folder = os.listdir(fpath2+year_month+"/")
                if np.isin(fname3,folder):
                    fnames3.append(fpath2+year_month+"/"+fname3)
                    fnames6.append(fpath2+year_month+"/"+fname6)
                    fnames5.append(fpath4+fname5)

    ds2 = read_data(fnames2,BOUNDARY,temporal_resolution,spatial_resolution)
    ds2['time'] = ds2['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    data_selected = xr.decode_cf(ds2,decode_times = True)['vMSE_col_adj']

    ds6   = read_data(fnames6,BOUNDARY,temporal_resolution,spatial_resolution) 
    ds6 = ds6.sel(level = 500)
    U850 = xr.decode_cf(ds6,decode_times = True)['U']

    ds3   = read_data(fnames3,BOUNDARY,temporal_resolution,spatial_resolution) 
    ds3 = ds3.sel(level = 500)
    V850 = xr.decode_cf(ds3,decode_times = True)['V']

    ds4 = read_data(fnames4,BOUNDARY,temporal_resolution,spatial_resolution)
    ds4['time'] = ds4['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    uMSE_adjust = xr.decode_cf(ds4,decode_times = True)['uMSE']
    vMSE_adjust = xr.decode_cf(ds4,decode_times = True)['vMSE']

    ds5 = read_data(fnames5,BOUNDARY,temporal_resolution,spatial_resolution)
    ds5 = ds5.sel(level = 500)
    MSE850 = xr.decode_cf(ds5,decode_times = True)['MSE']
    
    Data3 = U850-uMSE_adjust/MSE850
    Data4 = V850-vMSE_adjust/MSE850
    Data5 = MSE850
    Data4 = Data4-Data4.mean("time")
    Data4 = Data4-Data4.mean("longitude")
    Data5 = Data5-Data5.mean("time")
    Data5 = Data5-Data5.mean("longitude")

    # Create a mask: True where we want to keep data, False elsewhere
    
    #data_selected[...,260:] = 0
    #data_selected[...,0:120] = 0
    
    #data_selected[...,178:] = 0
    #data_selected[...,:177] = 0
    #data_selected[...,357:] = 0
    #data_selected[...,0:50] = 0
    #data_selected[...,0:35,...] = 0
    Data_selected = data_selected.sel(time = data_selected.time.dt.month.isin(months))
    Data_selected3 = Data3.sel(time = Data3.time.dt.month.isin(months))
    Data_selected4 = Data4.sel(time = Data4.time.dt.month.isin(months))
    Data_selected5 = Data5.sel(time = Data5.time.dt.month.isin(months))
    #Data[:,0:15,:] = -1*Data[:,0:15,:]

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
                'dosymmetries': False, 
                'rmvLowFrq':True}
    #symComponent, asymComponent = wf_analysis(Data_selected,Data_selected, **opt)

    # in this example, the smoothing & normalization will happen and use defaults
    power = wf_analysis(Data_selected4,Data_selected5, **opt)
    symComponent = power.integrate("frequency").integrate("wavenumber")
    
    symComponent = symComponent*2*np.pi*a*np.cos(symComponent.latitude/180*np.pi)*1e-15
    freq_low = 1.0/(2*24)
    freq_hi = 999#1.0/(2*24)
    wavenumber_low = -20.0
    wavenumber_hi = 0#-3.0

    #symComponent1 = power_mask(power,wavenumber_low,-1,freq_low,freq_hi).integrate("frequency").integrate("wavenumber")
    #symComponent1 = symComponent1*2*np.pi*a*np.cos(symComponent1.latitude/180*np.pi)*1e-15
#
    #symComponent2 = power_mask(power,wavenumber_low,-3,freq_low,freq_hi).integrate("frequency").integrate("wavenumber")
    #symComponent2 = symComponent2*2*np.pi*a*np.cos(symComponent2.latitude/180*np.pi)*1e-15
#
    #symComponent3 = power_mask(power,wavenumber_low,-5,freq_low,freq_hi).integrate("frequency").integrate("wavenumber")
    #symComponent3 = symComponent3*2*np.pi*a*np.cos(symComponent3.latitude/180*np.pi)*1e-15
    symComponent1 = power_mask(power,-999,0,0,1.0/(10*24)).integrate("frequency").integrate("wavenumber")
    symComponent1 = symComponent1*2*np.pi*a*np.cos(symComponent1.latitude/180*np.pi)*1e-15

    symComponent2 = power_mask(power,-999,0,1.0/(10*24),1.0/(2*24)).integrate("frequency").integrate("wavenumber")
    symComponent2 = symComponent2*2*np.pi*a*np.cos(symComponent2.latitude/180*np.pi)*1e-15

    symComponent3 = power_mask(power,-999,0,1.0/(2*24),999).integrate("frequency").integrate("wavenumber")
    symComponent3 = symComponent3*2*np.pi*a*np.cos(symComponent3.latitude/180*np.pi)*1e-15

    

    mask_south = symComponent.latitude < abs(symComponent).idxmin()
    mask_north = symComponent.latitude > abs(symComponent).idxmin()
    x = symComponent.latitude
    y1 = symComponent1/symComponent*100
    y2 = symComponent2/symComponent*100
    y3 = symComponent3/symComponent*100
    fig, ax = plt.subplots()
    #plt.plot(x[mask_south],y1[mask_south],"r", label = r"$1 \leqslant -k \leqslant 20$")
    #plt.plot(x[mask_north],y1[mask_north],"r")
    #plt.plot(x[mask_south],y2[mask_south],"b", label = r"$3 \leqslant -k \leqslant 20$")
    #plt.plot(x[mask_north],y2[mask_north],"b")
    #plt.plot(x[mask_south],y3[mask_south],"grey", label = r"$5 \leqslant -k \leqslant 20$")
    #plt.plot(x[mask_north],y3[mask_north],"grey")

    plt.plot(x[mask_south],y1[mask_south],"r", label = r"$0 \leqslant f \leqslant 1/10$")
    plt.plot(x[mask_north],y1[mask_north],"r")
    plt.plot(x[mask_south],y2[mask_south],"b", label = r"$1/10 \leqslant f \leqslant 1/2$")
    plt.plot(x[mask_north],y2[mask_north],"b")
    plt.plot(x[mask_south],y3[mask_south],"grey", label = r"$1/2 \leqslant f \leqslant 999$")
    plt.plot(x[mask_north],y3[mask_north],"grey")
    #ax.fill_between(x[mask_south],y1[mask_south], y2[mask_south], alpha=0.2,color = "r" )
    #ax.fill_between(x[mask_north],y1[mask_north], y2[mask_north], alpha=0.2,color = "r" )
    #ax.fill_between(x[mask_south],y2[mask_south], y3[mask_south], alpha=0.2,color = "b" )
    #ax.fill_between(x[mask_north],y2[mask_north], y3[mask_north], alpha=0.2,color = "b" )
    #ax.fill_between(x[mask_south],y3[mask_south], y3[mask_south]*0, alpha=0.2,color = "grey" )
    #ax.fill_between(x[mask_north],y3[mask_north], y3[mask_north]*0, alpha=0.2,color = "grey" )
    plt.ylabel("%",fontsize=15)
    plt.xlabel("Latitude",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)

    xticks = ax.get_xticks()
    # Format them: positive → 'N', negative → 'S', zero stays '0'
    xtick_labels = [f"{abs(int(t))}N" if t > 0 
                    else f"{abs(int(t))}S" if t < 0 
                    else "0" 
                    for t in xticks]
    
    ax.set_xticks(xticks)
    ax.set_xticklabels(xtick_labels)

    plt.xlim(min(symComponent.latitude),max(symComponent.latitude))
    #plt.ylim(0,100)
    plt.grid(linestyle = ':', linewidth = 0.5)
    plt.tick_params(axis="both",direction="in")
    ax.legend(loc="upper left")
    plt.savefig("WK_Ano_vMSE_cospectrum_verify_ratio_-20_-3.pdf")
    logging.info("figure 2 ploted")

    #outPlotName = "WK_Ano_vMSE_cospectrum_verify.pdf"
    #plt.figure()
    ##Data_selected6 = Data_selected6/Data_selected6.mean()*symComponent.mean()
    #aaa = xr.cov(Data_selected6, symComponent)/symComponent.var()
    #bbb = Data_selected6.mean()-aaa*symComponent.mean()
    #ccc = xr.cov(Data_selected6, symComponent)/(symComponent.std()*Data_selected6.std())
    #logging.info(aaa)
    #logging.info(bbb)
    #logging.info(aaa.values)
    #logging.info(bbb.values)
    #logging.info(ccc.values)
    #symComponent = symComponent*aaa+bbb
    #plt.plot(symComponent.latitude,symComponent,"b")
    #plt.plot(Data_selected6.latitude,Data_selected6.mean("time").mean("longitude"),":r")
    #plt.ylabel(r"$(PW)$",fontsize=15)
    #plt.xlabel("Latitude",fontsize=15)
    #plt.xticks(fontsize=14)
    #plt.yticks(fontsize=14)
    #plt.xlim(min(symComponent.latitude),max(symComponent.latitude))
    #plt.grid(linestyle = ':', linewidth = 0.5)
    #plt.tick_params(axis="both",direction="in")
    #plt.savefig(outPlotName)

    #outPlotName = "WK_Ano_vMSE_cospectrum_verify1.pdf"
    #plt.figure()
    #plt.plot(symComponent.latitude,symComponent,"b")
    #plt.ylabel(r"$(PW)$",fontsize=15)
    #plt.xlabel("Latitude",fontsize=15)
    #plt.xticks(fontsize=14)
    #plt.yticks(fontsize=14)
    #plt.xlim(min(symComponent.latitude),max(symComponent.latitude))
    #plt.grid(linestyle = ':', linewidth = 0.5)
    #plt.tick_params(axis="both",direction="in")
    #plt.savefig(outPlotName)
#
    #outPlotName = "WK_Ano_vMSE_cospectrum_verify2.pdf"
    #plt.figure()
    #plt.plot(Data_selected6.latitude,Data_selected6.mean("time").mean("longitude"),":r")
    #plt.ylabel(r"$(PW)$",fontsize=15)
    #plt.xlabel("Latitude",fontsize=15)
    #plt.xticks(fontsize=14)
    #plt.yticks(fontsize=14)
    #plt.xlim(min(symComponent.latitude),max(symComponent.latitude))
    #plt.grid(linestyle = ':', linewidth = 0.5)
    #plt.tick_params(axis="both",direction="in")
    #plt.savefig(outPlotName)


