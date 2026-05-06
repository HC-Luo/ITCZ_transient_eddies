#cospectrum analysis of v and MSE anomalies, latitude vs phase speed
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
        z_final = 2.0 * z_pee.mean(dim='time').squeeze()
    return z_final

def wf_analysis(x1,x2, **kwargs):
    z2 = spacetime_power(x1,x2, **kwargs)#.mean("component")

    # Constants
    a = 6.371e6            # Earth's radius in meters
    seconds_per_day = 86400
    
    # Get coordinate values
    lat = z2.coords['latitude']
    k = z2.coords['wavenumber']
    omega = z2.coords['frequency']  # in cpd (cycles per day)
    
    # Create 2D (k, omega) meshgrid
    K, OMEGA = np.meshgrid(k, omega, indexing='ij')  # shape (n_k, n_omega)
    
    # Define phase speed bins (in m/s)
    phase_speed_bins = np.linspace(-100, 100, 201)
    phase_speed_centers = 0.5 * (phase_speed_bins[:-1] + phase_speed_bins[1:])
    
    # Initialize output
    phase_speed_spectrum = xr.DataArray(
        np.zeros((len(lat), len(phase_speed_centers))),
        coords={'latitude': lat, 'phase_speed': phase_speed_centers},
        dims=('latitude', 'phase_speed')
    )
    
    # Compute and bin for each latitude
    for i, phi_deg in enumerate(lat.values):
        phi_rad = np.deg2rad(phi_deg)
        circumference = 2 * np.pi * a * np.cos(phi_rad)  # meters
        phase_speed = (OMEGA / K) * (circumference / seconds_per_day)  # m/s
    
        # Extract spectrum slice
        spec = z2.sel(latitude=phi_deg).values  # shape (k, omega)
        
        # Mask where k == 0 to avoid divide-by-zero
        valid_mask = np.isfinite(phase_speed) & np.isfinite(spec) & (K != 0)
        
        # Flatten
        flat_c = phase_speed[valid_mask]
        flat_power = spec[valid_mask]
    
        # Bin using np.histogram
        binned_power, _ = np.histogram(
            flat_c,
            bins=phase_speed_bins,
            weights=flat_power
        )
    
        # Store result
        phase_speed_spectrum[i, :] = binned_power


    
    return phase_speed_spectrum


if __name__ == "__main__":
    fpath1 = "/glade/derecho/scratch/hcluo/"
    fpath2 = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/"
    fpath3 = "/glade/work/hcluo/data/MSE_adjust/"
    fpath4 = "/glade/derecho/scratch/hcluo/MSE/"
    years = np.arange(1997,2024)
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    MONTHS = ["%02d" % x for x in np.arange(1,13)]
    days = ["%02d" % x for x in np.arange(1,32)]
    BASIN = "EI"
    BOUNDARY = [-20,20,0,360]
    if BASIN == "EP":
        BOUNDARY1 = [-10,30,-150,-75]
        BOUNDARY2 = [5,20,-125,-90]
    elif BASIN == "EI":
        BOUNDARY1 = [0,30,60,100]
        BOUNDARY2 = [15,20,85,90]
    elif BASIN == "NA":
        BOUNDARY1 = [-10,30,-40,15]
        BOUNDARY2 = [5,20,-27.5,10]
    elif BASIN == "SA":
        BOUNDARY1 = [-30,10,-100,-25]
        BOUNDARY2 = [-20,-3,-72,-45]
    elif BASIN == "WP":
        BOUNDARY1 = [-10,30,100,180]
        BOUNDARY2 = [5,20,120,150]
    spatial_resolution = 0.5 #degree
    temporal_resolution = 12 #hrs, can only be 1, 2, 3, 4, 6, 12
    a = 6.371e6  
    #fnames1 = []
    fnames2 = []
    fnames3 = []
    fnames4 = []
    fnames5 = []
    fnames6 = []
    for i in years:
        fname2 = "vMSE_col_transient/vMSE_col_transient_"+f"{i}.nc"
        fnames2.append(fpath1+fname2)
        for j in np.arange(1,13):
            year_month = str(i)+MONTHS[j-1]
            fname4 = "adjustment_"+year_month+".nc"
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
    data_selected = xr.decode_cf(ds2,decode_times = True)['vMSE_transient_col']

    ds6   = read_data(fnames6,BOUNDARY,temporal_resolution,spatial_resolution) 
    ds6 = ds6.sel(level = 800)
    U800 = xr.decode_cf(ds6,decode_times = True)['U']

    ds3   = read_data(fnames3,BOUNDARY,temporal_resolution,spatial_resolution) 
    ds3 = ds3.sel(level = 800)
    V800 = xr.decode_cf(ds3,decode_times = True)['V']

    ds4 = read_data(fnames4,BOUNDARY,temporal_resolution,spatial_resolution)
    ds4['time'] = ds4['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    uMSE_adjust = xr.decode_cf(ds4,decode_times = True)['uMSE']
    vMSE_adjust = xr.decode_cf(ds4,decode_times = True)['vMSE']

    ds5 = read_data(fnames5,BOUNDARY,temporal_resolution,spatial_resolution)
    ds5 = ds5.sel(level = 800)
    MSE800 = xr.decode_cf(ds5,decode_times = True)['MSE']
    
    Data3 = U800-uMSE_adjust/MSE800
    Data4 = V800-vMSE_adjust/MSE800
    Data5 = MSE800
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
    #Data_selected = data_selected.sel(time = data_selected.time.dt.month.isin(months))
    #Data3[...,0:210] = 0
    #Data3[...,271:] = 0
    #Data4[...,0:210] = 0
    #Data4[...,271:] = 0
    #Data5[...,0:210] = 0
    #Data5[...,271:] = 0
    #Data3[...,0:60] = 0
    #Data3[...,101:] = 0
    #Data4[...,0:60] = 0
    #Data4[...,101:] = 0
    #Data5[...,0:60] = 0
    #Data5[...,101:] = 0

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
    # in this example, the smoothing & normalization will happen and use defaults
    #phase_speed_spectrum = wf_analysis(Data_selected, **opt)
    phase_speed_spectrum = wf_analysis(Data_selected4,Data_selected5, **opt)

    # Extract data and coordinates
    lat = phase_speed_spectrum.latitude
    c = phase_speed_spectrum.phase_speed
    Z = phase_speed_spectrum

    # significance test
    #n = 2*len(Data_selected4.latitude)*len(Data_selected4.time)/spd/nDayWin
    #chi_square = chi2.ppf(0.95, n, loc=0, scale=1)
    #background = copy.deepcopy(Z)
    #background = smooth_1_2_1(background, 'phase_speed', 10)
    #background = smooth_1_2_1(background, 'wavenumber', 40)
    #rc = chi_square/(n-1)*background
    #Z = Z.where(Z > rc, np.nan)
    
    # Mask near-zero phase speeds
    threshold = 1e-2  # m/s
    # Create boolean masks for eastward and westward phase speeds
    mask_east = c > threshold
    mask_west = c < -threshold
    
    # Apply mask to c and Z along phase speed dimension (axis 1)
    c_east = c[mask_east]              # (e.g. 100,)
    Z_east = Z[:, mask_east]           # (41, n_east)
    
    c_west = c[mask_west]              # (e.g. 90,)
    Z_west = Z[:, mask_west]           # (41, n_west)
    
    # Plot
    fig, ax = plt.subplots(figsize=(5, 4))
    cmap = wf.colormap(color="BluWhiRed")
    # Westward contours (blue)
    # Define contour levels spaced by 1e12
    min_val = np.floor(Z.min() / 1e-2) * 1e-2
    max_val = np.ceil(Z.max() / 1e-2) * 1e-2
    #levels = [x for x in np.arange(min_val, max_val + 1e2, 1e2) if x!=0]
    levels = np.linspace(-600,600,13)
    cs1 = plt.contourf(c_west, lat, Z_west, 
                      levels=levels, 
                      extend = 'both',
                      cmap = cmap)
    #plt.clabel(cs1, fmt='%1.1f', inline=True, fontsize=8)
    
    # Eastward contours (red)
    cs2 = ax.contourf(c_east, lat, Z_east, 
                      levels=levels, 
                      extend = 'both',
                      cmap = cmap)
    #plt.clabel(cs2, fmt='%1.1f', inline=True, fontsize=8)    
    ax.plot(Data_selected3.mean(["time","longitude"]),Data_selected3.latitude)
    #plt.plot(Data4.mean(["time","longitude"]),Data4.latitude)
    cbar = plt.colorbar(cs2,
                       extendrect = True,
                       extendfrac = 'auto',
                       orientation='vertical',
                       #ticks=lb_ticks,
                       shrink=0.8,
                       aspect = 30,
                       drawedges=False)#,
                       #pad=0.04)
                                 
    plt.axvline(0, color='gray', linestyle='--', linewidth=1)  # show c = 0
    #plt.grid(True)
    plt.xlabel("Phase Speed (m/s)")
    plt.ylabel("Latitude")
    plt.xlim([-20,20])
    # Get current yticks
    yticks = ax.get_yticks()
    
    # Format them: positive → 'N', negative → 'S', zero stays '0'
    ytick_labels = [f"{abs(int(t))}N" if t > 0 
                    else f"{abs(int(t))}S" if t < 0 
                    else "0" 
                    for t in yticks]
    
    ax.set_yticks(yticks)
    ax.set_yticklabels(ytick_labels)
    plt.tick_params(axis='both', direction='in')
    plt.tight_layout()
    plt.savefig(f"SP_Ano_vMSE_cospectrum_800.pdf")

