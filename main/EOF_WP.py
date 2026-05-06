#EOF analysis on TD waves at Western Pacific
import numpy as np 
import xarray as xr
import myfun as wf
import math, gc, logging,os
import matplotlib as mpl
import matplotlib.pyplot as plt
logging.getLogger('matplotlib.font_manager').disabled = True
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import geocat.viz as gv
from scipy.signal import convolve
import xeofs as xe
import myfun as mf
from matplotlib.ticker import MaxNLocator

def main():
    BASIN = "WP"
    years = np.arange(1997,2024)
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    fout0 =      f"EOF_sigtest_{BASIN}.pdf"
    fout1 =     f"EOF_div_vMSE_{BASIN}.pdf"
    fout2 = f"EOF_Ano_vMSE_{BASIN}.pdf"

    MONTHS = ["%02d" % x for x in np.arange(1,13)]
    days = ["%02d" % x for x in np.arange(1,32)]
    BOUNDARY  = [-30,30,0,360]
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
    
    spatial_resolution = 0.5#degree
    spatial_resolution2 = 1.0 #degree
    temporal_resolution = 12 #hrs, can only be 1, 2, 3, 4, 6, 12
    temporal_resolution2 = 1 #days
    freq_low = 1/(10*24)
    freq_hi = 1/(2*24)
    wavenumber_low = -20
    wavenumber_hi = -5
    
    fpath1 =  "/glade/campaign/collections/rda/data/d633000/e5.oper.fc.sfc.meanflux/"
    fpath2 = "/glade/derecho/scratch/hcluo/"
    fpath3 = "/glade/campaign/collections/rda/data/d728007/gpcp_v1.3_daily/"
    fpath4 = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/"
    fpath5 = "/glade/work/hcluo/data/MSE_adjust/"
    fpath6 = "/glade/derecho/scratch/hcluo/MSE/"

    fnames1 = []
    fnames2 = []
    fnames3 = []
    fnames4 = []
    fnames5 = []
    fnames6 = []
    fnames7 = []
    fnames8 = []

    for i in years:
        fname1 = "vMSE_col/div_vMSE_col_"+f"{i}.nc"
        fnames1.append(fpath2+fname1)

        fname4 = "vMSE_col_transient/vMSE_col_transient_"+f"{i}.nc"
        fnames4.append(fpath2+fname4)

        for j in np.arange(1,13):
            year_month = str(i)+MONTHS[j-1]
            fname7 = "adjustment_"+year_month+".nc"
            fnames7.append(fpath5+fname7)
            
            for k in days:
                fname3 = "gpcp_v01r03_daily_d"+year_month+k+".nc"
                folder3 = os.listdir(fpath3+str(i)+"/")
                if np.isin(fname3,folder3):
                    fnames3.append(fpath3+str(i)+"/"+fname3)

                fname5 = "e5.oper.an.pl.128_131_u.ll025uv." \
                +year_month+k+"00_"+year_month+k+"23.nc"
                fname6 = "e5.oper.an.pl.128_132_v.ll025uv." \
                +year_month+k+"00_"+year_month+k+"23.nc"
                fname8 = "MSE_"+year_month+k+"00_"+year_month+k+"18.nc"    
                folder = os.listdir(fpath4+year_month+"/")
                if np.isin(fname5,folder):
                    fnames5.append(fpath4+year_month+"/"+fname5)
                    fnames6.append(fpath4+year_month+"/"+fname6)
                    fnames8.append(fpath6+fname8)

    ds = read_data2(fpath1,years,MONTHS,BOUNDARY,spatial_resolution,temporal_resolution)
    ds["longitude"] = (ds["longitude"] + 180) % 360 - 180
    ds = ds.sortby('longitude')
    ds = ds.sel(longitude=slice(BOUNDARY1[2], BOUNDARY1[3]))
    Data1 = -xr.decode_cf(ds,decode_times = True)['OLR']
    
    ds1 = read_data(fnames1,BOUNDARY,temporal_resolution,spatial_resolution)
    ds1["longitude"] = (ds1["longitude"] + 180) % 360 - 180
    ds1 = ds1.sortby('longitude')
    ds1['time'] = ds1['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data2 = xr.decode_cf(ds1,decode_times = True)['div_vMSE_col']
    Data7 = Data2.chunk(dict(longitude=-1))
    Data2 = Data2.sel(longitude=slice(BOUNDARY1[2], BOUNDARY1[3]))

    Data7 = Data7.rolling(longitude=5, min_periods = 1,center=True).mean().dropna("longitude")
    Data7 = Data7.sel(time = Data7.time.dt.month.isin(months)).mean("time")
    Data7 = Data7.sel(longitude=slice(BOUNDARY1[2], BOUNDARY1[3]))

    ds3 = read_data(fnames3,BOUNDARY,temporal_resolution2,spatial_resolution2)
    ds3["longitude"] = (ds3["longitude"] + 180) % 360 - 180
    ds3 = ds3.sortby('longitude')
    Data3 = xr.decode_cf(ds3,decode_times = True)['precip']
    Data9 = precip_centroid(Data3.sel(time = Data3.time.dt.month.isin(months))).mean("time")
    Data3 = Data3.sel(longitude=slice(BOUNDARY1[2], BOUNDARY1[3]))

    Data9 = Data9.rolling(longitude=5, min_periods = 1,center=True).mean().dropna("longitude")
    Data9 = Data9.sel(longitude=slice(BOUNDARY1[2], BOUNDARY1[3]))
    
    ds4 = read_data(fnames4,BOUNDARY,temporal_resolution,spatial_resolution)
    ds4["longitude"] = (ds4["longitude"] + 180) % 360 - 180
    ds4 = ds4.sortby('longitude')
    ds4['time'] = ds4['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data4 = xr.decode_cf(ds4,decode_times = True)['vMSE_transient_col']
    Data8 = Data4.chunk(dict(longitude=-1))
    Data4 = Data4.sel(longitude=slice(BOUNDARY1[2], BOUNDARY1[3]))

    Data8 = Data8.rolling(longitude=5, min_periods = 1,center=True).mean().dropna("longitude")
    Data8 = Data8.sel(time = Data8.time.dt.month.isin(months)).mean("time")
    Data8 = Data8.sel(longitude=slice(BOUNDARY1[2], BOUNDARY1[3]))

    ds5   = read_data(fnames5,BOUNDARY,temporal_resolution,spatial_resolution) 
    ds5["longitude"] = (ds5["longitude"] + 180) % 360 - 180
    ds5 = ds5.sortby('longitude')
    ds5 = ds5.sel(level = 850,longitude=slice(BOUNDARY1[2], BOUNDARY1[3]))
    U850 = xr.decode_cf(ds5,decode_times = True)['U']

    ds6   = read_data(fnames6,BOUNDARY,temporal_resolution,spatial_resolution) 
    ds6["longitude"] = (ds6["longitude"] + 180) % 360 - 180
    ds6 = ds6.sortby('longitude')
    ds6 = ds6.sel(level = 850,longitude=slice(BOUNDARY1[2], BOUNDARY1[3]))
    V850 = xr.decode_cf(ds6,decode_times = True)['V'] 

    ds7 = read_data(fnames7,BOUNDARY,temporal_resolution,spatial_resolution)
    ds7['time'] = ds7['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    ds7["longitude"] = (ds7["longitude"] + 180) % 360 - 180
    uMSE_adjust = xr.decode_cf(ds7,decode_times = True)['uMSE']
    vMSE_adjust = xr.decode_cf(ds7,decode_times = True)['vMSE']

    ds8 = read_data(fnames8,BOUNDARY,temporal_resolution,spatial_resolution)
    ds8["longitude"] = (ds8["longitude"] + 180) % 360 - 180
    ds8 = ds8.sel(level = 850)
    MSE850 = xr.decode_cf(ds8,decode_times = True)['MSE']
    
    Data5 = U850-uMSE_adjust/MSE850
    Data6 = V850-vMSE_adjust/MSE850





    # Apply the Lanczos band-pass filter
    #Data1_filtered = lanczos_bandpass_filter(
    #    Data1.chunk(dict(time=-1)),
    #    low_cutoff_days=10,   # 6-day cutoff (longer period)
    #    high_cutoff_days=1,  # 2-day cutoff (shorter period)
    #    m=60                 # Number of lobes (adjust based on desired roll-off)
    #)
    
    Data1_filtered = fft_filter(
        Data1,
        time_low=freq_low,  # 1/30 cycles/h ≈ 30-h period
        time_high=freq_hi,   # 1/5 cycles/h ≈ 5-h period
        lon_low=wavenumber_low,    # 1/20 cycles/degree ≈ 20-degree wavelength
        lon_high=wavenumber_hi     # 1/5 cycles/degree ≈ 5-degree wavelength
    )
    
    
    Data2_filtered = fft_filter(
        Data2.chunk(dict(time=-1)),
        time_low=freq_low,  # 1/30 cycles/h ≈ 30-h period
        time_high=freq_hi,   # 1/5 cycles/h ≈ 5-h period
        lon_low=wavenumber_low,    # 1/20 cycles/degree ≈ 20-degree wavelength
        lon_high=wavenumber_hi     # 1/5 cycles/degree ≈ 5-degree wavelength
    )
    
    
    Data4_filtered = fft_filter(
        Data4.chunk(dict(time=-1)),
        time_low=freq_low,  # 1/30 cycles/h ≈ 30-h period
        time_high=freq_hi,   # 1/5 cycles/h ≈ 5-h period
        lon_low=wavenumber_low,    # 1/20 cycles/degree ≈ 20-degree wavelength
        lon_high=wavenumber_hi     # 1/5 cycles/degree ≈ 5-degree wavelength
    )

    Data5_filtered = fft_filter(
        Data5.chunk(dict(time=-1)),
        time_low=freq_low,  # 1/30 cycles/h ≈ 30-h period
        time_high=freq_hi,   # 1/5 cycles/h ≈ 5-h period
        lon_low=wavenumber_low,    # 1/20 cycles/degree ≈ 20-degree wavelength
        lon_high=wavenumber_hi     # 1/5 cycles/degree ≈ 5-degree wavelength
    )
    Data6_filtered = fft_filter(
        Data6.chunk(dict(time=-1)),
        time_low=freq_low,  # 1/30 cycles/h ≈ 30-h period
        time_high=freq_hi,   # 1/5 cycles/h ≈ 5-h period
        lon_low=wavenumber_low,    # 1/20 cycles/degree ≈ 20-degree wavelength
        lon_high=wavenumber_hi     # 1/5 cycles/degree ≈ 5-degree wavelength
    )

    Data1_filtered = Data1_filtered.sel(time = Data1_filtered.time.dt.month.isin(months))
    Data2_filtered = Data2_filtered.sel(time = Data2_filtered.time.dt.month.isin(months))
    Data4_filtered = Data4_filtered.sel(time = Data4_filtered.time.dt.month.isin(months))
    Data5_filtered = Data5_filtered.sel(time = Data5_filtered.time.dt.month.isin(months))
    Data6_filtered = Data6_filtered.sel(time = Data6_filtered.time.dt.month.isin(months))

    Data1_filtered = Data1_filtered - Data1_filtered.mean("time")
    Data2_filtered = Data2_filtered - Data2_filtered.mean("time")
    Data4_filtered = Data4_filtered - Data4_filtered.mean("time")
    Data5_filtered = Data5_filtered - Data5_filtered.mean("time")
    Data6_filtered = Data6_filtered - Data6_filtered.mean("time")

    scores, components, expvar = EOF(Data1_filtered.sel(latitude = slice(BOUNDARY2[0],BOUNDARY2[1]),longitude = slice(BOUNDARY2[2],BOUNDARY2[3])),
                                     test = True, fout = fout0)
    regression1 = xr.DataArray(data=np.zeros((3, len(Data1_filtered.latitude), len(Data1_filtered.longitude))),
                               coords={"mode": np.arange(1,4),
                                       "latitude": Data1_filtered.latitude,
                                       "longitude": Data1_filtered.longitude})
    
    regression2 = xr.DataArray(data=np.zeros((3, len(Data2_filtered.latitude), len(Data2_filtered.longitude))),
                               coords={"mode": np.arange(1,4),
                                       "latitude": Data2_filtered.latitude,
                                       "longitude": Data2_filtered.longitude})
    regression4 = xr.DataArray(data=np.zeros((3, len(Data4_filtered.latitude), len(Data4_filtered.longitude))),
                               coords={"mode": np.arange(1,4),
                                       "latitude": Data4_filtered.latitude,
                                       "longitude": Data4_filtered.longitude})
    
    regression5 = xr.DataArray(data=np.zeros((3, len(Data5_filtered.latitude), len(Data5_filtered.longitude))),
                               coords={"mode": np.arange(1,4),
                                       "latitude": Data5_filtered.latitude,
                                       "longitude": Data5_filtered.longitude})
    regression6 = xr.DataArray(data=np.zeros((3, len(Data6_filtered.latitude), len(Data6_filtered.longitude))),
                               coords={"mode": np.arange(1,4),
                                       "latitude": Data6_filtered.latitude,
                                       "longitude": Data6_filtered.longitude})

    for i in range(3):
        regression1[i,...] = mf.regression(scores[i,...],Data1_filtered,test = True,rc = 0.036)
        regression2[i,...] = mf.regression(scores[i,...],Data2_filtered,test = True,rc = 0.036)
        regression4[i,...] = mf.regression(scores[i,...],Data4_filtered,test = True,rc = 0.036)
        regression5[i,...] = mf.regression(scores[i,...],Data5_filtered,test = True,rc = 0.036)
        regression6[i,...] = mf.regression(scores[i,...],Data6_filtered,test = True,rc = 0.036)
    
    EOF_plot(regression1,regression2/1.0e6,regression5,regression6,
             Data7,Data9,expvar,BOUNDARY1,cmap = wf.colormap(color="BluWhiRed"),
             title = r"$\langle vh \rangle_D \;(\times 10^6 \; W \, m^{-1})$",
             fout = fout1)
    EOF_plot(regression1,regression4/1.0e6,regression5,regression6,
             Data8,Data9,expvar,BOUNDARY1,cmap = wf.colormap(color="BluWhiRed"),
             title = r"$\langle v^*' h^*' \rangle \;(\times 10^6 \; W \, m^{-1})$",
             fout = fout2)
    return

def read_data(fnames,BOUNDARY,temporal_resolution,spatial_resolution):
    ds = xr.open_mfdataset(fnames,concat_dim = 'time',combine='nested',decode_times=False)
    builtin_spatial_resolution = (ds.longitude[1]-ds.longitude[0]).values
    if (len(ds.latitude) > 1) & (ds.latitude[0] > ds.latitude[1]):
        ds = ds.reindex(latitude=ds.latitude[::-1])

    if "time" in ds.coords:
        TIME = ds['time']
        builtin_temporal_resolution = (TIME[1]-TIME[0]).values

        ds = ds.sel(time = slice(TIME[0],TIME[-1],int(temporal_resolution/builtin_temporal_resolution)),
                    latitude = slice(min(BOUNDARY[0],BOUNDARY[1]),max(BOUNDARY[0],BOUNDARY[1]),int(spatial_resolution/builtin_spatial_resolution)), 
                    longitude = slice(min(BOUNDARY[2],BOUNDARY[3]),max(BOUNDARY[2],BOUNDARY[3]),int(spatial_resolution/builtin_spatial_resolution)))

    else: 
        ds = ds.sel(latitude = slice(min(BOUNDARY[0],BOUNDARY[1]),max(BOUNDARY[0],BOUNDARY[1]),int(spatial_resolution/builtin_spatial_resolution)), 
                    longitude = slice(min(BOUNDARY[2],BOUNDARY[3]),max(BOUNDARY[2],BOUNDARY[3]),int(spatial_resolution/builtin_spatial_resolution)))

    #if "level" in ds.coords:
    #    ds = ds.sel(level = slice(100,1000))

    return ds

def read_data2(fpath, years, months, BOUNDARY, spatial_resolution, temporal_resolution):
    fnames = []
    for i in years:
        for j in range(len(months)):
            fname1 = fpath+str(i)+months[j]+"/e5.oper.fc.sfc.meanflux.235_040_mtnlwrf.ll025sc."\
            +str(i)+months[j]+"0106_"+str(i)+months[j]+"1606.nc"
            
            fnames.append(fname1)
    
    
            if months[j] == '12':
                fname2 = fpath+str(i)+"12/e5.oper.fc.sfc.meanflux.235_040_mtnlwrf.ll025sc."\
                +str(i)+"121606_"+str(i+1)+"010106.nc"
            else:
                fname2 = fpath+str(i)+months[j]+"/e5.oper.fc.sfc.meanflux.235_040_mtnlwrf.ll025sc."\
                +str(i)+months[j]+"1606_"+str(i)+months[j+1]+"0106.nc"
    
            fnames.append(fname2)
    
    ds = xr.open_mfdataset(fnames,concat_dim = 'forecast_initial_time', combine='nested',decode_times=False)
    
    #slicing and regridding
    if (len(ds.latitude) > 1) & (ds.latitude[0] > ds.latitude[1]):
        ds = ds.reindex(latitude=ds.latitude[::-1])
    builtin_resolution = (ds.longitude[1]-ds.longitude[0]).values
    ds = ds.sel(forecast_hour = slice(temporal_resolution,12,temporal_resolution),latitude = slice(min(BOUNDARY[0],BOUNDARY[1]),max(BOUNDARY[0],BOUNDARY[1]),int(spatial_resolution/builtin_resolution)), longitude = slice(min(BOUNDARY[2],BOUNDARY[3]),max(BOUNDARY[2],BOUNDARY[3]),int(spatial_resolution/builtin_resolution)))
    
    de = ds.stack(time=("forecast_initial_time","forecast_hour"))
    data = de['MTNLWRF']
    data = data.transpose("time", "latitude", "longitude")
    
    Data = xr.DataArray(data.values, dims=['time','latitude','longitude'], 
                            coords=dict(
                            latitude=data.latitude,
                            longitude=data.longitude,
                            time=ds.forecast_initial_time.values[0]+temporal_resolution*np.arange(1,1+len(data.time))))
    
    ds_out = xr.Dataset(data_vars = {"OLR":(("time","latitude","longitude"),Data.data)},
                                coords = {"time":Data.time,
                                          "latitude":Data.latitude,
                                          "longitude":Data.longitude})
    ds_out['time'] = ds_out['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    logging.info("Finishing reading data")
    return ds_out

#Lanczos band-pass filter
def lanczos_bandpass_filter(da, low_cutoff_days, high_cutoff_days, m=60):
    """
    Apply a Lanczos band-pass filter to an xarray DataArray.
    
    Parameters:
        da (xarray.DataArray): Input data with 'time', 'latitude', 'longitude' dimensions.
        low_cutoff_days (float): Long-period cutoff (days) for the band-pass filter.
        high_cutoff_days (float): Short-period cutoff (days) for the band-pass filter.
        m (int): Number of filter lobes, determining the filter length (2m+1 points).
        
    Returns:
        xarray.DataArray: Filtered data with the same dimensions as input.
    """
    # Calculate time step in days
    dt_days = (da.time[1] - da.time[0]).values / np.timedelta64(1, 'D')
    
    # Desired cutoff frequencies (cycles per day)
    f_low_desired = 1 / low_cutoff_days   # Lower frequency (longer period)
    f_high_desired = 1 / high_cutoff_days # Higher frequency (shorter period)
    
    # Convert to digital frequencies (cycles per sample)
    f_low = f_low_desired * dt_days
    f_high = f_high_desired * dt_days
    
    # Adjust high cutoff to avoid exceeding Nyquist frequency
    nyquist = 0.5
    if f_high >= nyquist:
        f_high = nyquist * 0.999
        print(f"Adjusted high cutoff to {f_high:.3f} cycles per sample.")
    
    # Generate filter coefficients
    n = np.arange(-m, m + 1)
    
    # Ideal band-pass filter coefficients
    h_ideal = (2 * f_high * np.sinc(2 * f_high * n)) - (2 * f_low * np.sinc(2 * f_low * n))
    
    # Lanczos window
    window = np.sinc(n / m)
    
    # Apply window to ideal coefficients
    h = h_ideal * window
    
    # Define filter application function
    def apply_filter(x):
        filtered = convolve(x, h, mode='same')
        filtered[:m] = np.nan  # Set edges to NaN
        filtered[-m:] = np.nan
        return filtered
    
    # Apply the filter along the time dimension
    da_filtered = xr.apply_ufunc(
        apply_filter,
        da,
        input_core_dims=[['time']],
        output_core_dims=[['time']],
        output_sizes={'time': da.sizes['time']},
        exclude_dims={'time'},
        vectorize=True,
        dask='parallelized',
        output_dtypes=[da.dtype]
    )
    da_filtered['time'] = da.time
    
    return da_filtered.transpose(*da.dims)

#WK-FFT band-pass filter
def fft_filter(da, time_low, time_high, lon_low, lon_high):
    """
    Apply a 2D bandpass filter using FFT on time and longitude for a 3D xarray DataArray.

    Parameters:
    da (xarray.DataArray): Input data with dimensions 'time', 'latitude', 'longitude'.
    time_low (float): Lower bound for temporal frequency (cycles/day).
    time_high (float): Upper bound for temporal frequency.
    lon_low (float): Lower bound for longitude wavenumber (cycles/degree).
    lon_high (float): Upper bound for longitude wavenumber.

    Returns:
    xarray.DataArray: Filtered data with the same dimensions.
    """
    # Ensure dimensions are ordered as time, latitude, longitude
    da = da.transpose('time', 'latitude', 'longitude')
    data = da.values
    n_time, _, n_lon = data.shape

    # Compute FFT along time (axis 0) and longitude (axis 2)
    fft_data = np.fft.fftn(data, axes=(0, 2))

    # Calculate temporal frequencies (cycles/day)
    time_step_days = (da.time[1] - da.time[0]).values / np.timedelta64(1, 'h')
    freqs = np.fft.fftfreq(n_time, time_step_days)

    # Calculate longitude wavenumbers (cycles/degree)
    lon_step = (da.longitude[1] - da.longitude[0]).values
    k_lon = 2*360*np.fft.fftfreq(n_lon, lon_step)
    #k_lon = 360*np.fft.fftfreq(n_lon, lon_step)##################################
    # Create 2D grid of frequencies and wavenumbers
    FREQ, K_LON = np.meshgrid(freqs, k_lon, indexing='ij')

    # Build 2D mask for time and longitude
    mask = ((
        (FREQ >= time_low) & (FREQ <= time_high) &
        (K_LON >= lon_low) & (K_LON <= lon_high)
    ) | (
        (FREQ <= -time_low) & (FREQ >= -time_high) &
        (K_LON <= -lon_low) & (K_LON >= -lon_high)
    ))

    # Reshape mask to 3D (broadcast over latitude)
    mask_3d = mask[:, np.newaxis, :]

    # Apply mask and inverse FFT
    fft_filtered = fft_data * mask_3d
    filtered_data = np.fft.ifftn(fft_filtered, axes=(0, 2)).real  # Keep real part

    # Reconstruct xarray DataArray
    filtered_da = xr.DataArray(
        filtered_data,
        dims=da.dims,
        coords=da.coords,
        attrs=da.attrs
    )
    return filtered_da

# precipitation centroid
def precip_centroid(Data):
    P = Data.sel(latitude = slice(-20,20))
    spatial_resolution = abs(P.longitude[1] - P.longitude[0])
    P_area = P*np.cos(P.latitude/180*np.pi)
    top_plus_bottom = (P_area+P_area.shift(latitude = 1))
    top_plus_bottom[:,0] = 0
    P_cent = (top_plus_bottom/2*spatial_resolution).cumsum("latitude")
    median = 0.5*(top_plus_bottom/2*spatial_resolution).sum("latitude")
    data = abs(P_cent-median).idxmin(dim = "latitude")
    return data

# EOF 
def EOF(var,test=False,**kwargs):
    model = xe.single.EOF(n_modes=5)
    model.fit(var, dim="time")
    expvar = model.explained_variance_ratio()
    components = model.components()
    scores = model.scores()

    if test:
        fout = kwargs.get('fout', 'EOF_sigtest.pdf')
        # Extract eigenvalues (λ) and get number of samples (N)
        eigenvalues = model.explained_variance()
        n_samples = len(var.time)
        errors = eigenvalues * np.sqrt(2 / n_samples)

        # Display results
        for i, (eig, err) in enumerate(zip(eigenvalues, errors), start=1):
            print(f"Mode {i}: λ = {eig:.4f} ± {err:.4f}")
            if i > 1:
                prev_eig, prev_err = eigenvalues[i-2], errors[i-2]
                # Check overlap with previous mode
                if (eig + err) >= (prev_eig - prev_err):
                    print(f"  -> Mode {i} is NOT distinct from Mode {i-1}")
                else:
                    print(f"  -> Mode {i} is distinct from Mode {i-1}")

        # Plot eigenvalues with error bars
        fig, ax = plt.subplots(figsize=(3, 2.5))
        mode_numbers = np.arange(1, len(eigenvalues) + 1)
        
        ax.errorbar(mode_numbers, eigenvalues, yerr=errors, fmt='o', capsize=5)
        ax.set_xlabel('Mode')
        ax.set_ylabel('Eigenvalue')
        #ax.set_title(kwargs.get('title', "North's Test"))
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.grid(True, linestyle='--', alpha=0.5)
        #ax.legend()
        
        plt.tight_layout()
        plt.savefig(fout)

    return scores, components, expvar

def EOF_plot(data1,data2,data3,data4,data5,data6,expvar,BOUNDARY,cmap,title,fout):
    modes = [1,2]#data1.mode.values
    if abs(BOUNDARY[3] - BOUNDARY[2]) > 50:
        step = 4
    else:
        step = 2
    fig, axes = plt.subplots(
        nrows=1, ncols=len(modes),
        subplot_kw={'projection': ccrs.PlateCarree()},
        figsize=(5 * len(modes), 5)
    )

    # Ensure axes is iterable
    if len(modes) == 1:
        axes = [axes]
    #x_ticks = np.arange(BOUNDARY[2]-360,BOUNDARY[3]-360,30)
    x_ticks = np.arange(BOUNDARY[2],BOUNDARY[3],30)
    y_ticks = np.arange(BOUNDARY[0],BOUNDARY[1]+10,10)
    x_labels = [f"{abs(lon)}W" if lon < 0 else (f"{lon}" if lon == 0 else f"{lon}E") for lon in x_ticks]
    y_labels = [f"{abs(lat)}S" if lat < 0 else (f"{lat}" if lat == 0 else f"{lat}N") for lat in y_ticks]
    
    levelmax1 = np.ceil(abs(data1.max()) + abs(data1.min()))/2
    levels1 = np.arange(-levelmax1, levelmax1+2, 2)

    if 2*abs(data2[0:len(modes),...]).mean() >= 3:
        levelmax2 = np.ceil(2*abs(data2[0:len(modes),...]).mean())
        levels2 = np.arange(-levelmax2, levelmax2+1, 1)
    else:
        levelmax2 = np.ceil(4*abs(data2[0:len(modes),...]).mean())/2
        levels2 = np.arange(-levelmax2, levelmax2+0.5, 0.5)

    # Loop over each mode and plot
    for i, mode in enumerate(modes):
        ax = axes[i]
        shadings = ax.contourf(data2.longitude, data2.latitude, data2.sel(mode = mode),cmap = cmap,levels = levels2,extend = 'both')
        contours =  ax.contour(data1.longitude, data1.latitude, data1.sel(mode = mode),colors = 'k',levels = levels1)
        vectors = ax.quiver(data3.longitude[::step], data3.latitude[::step], data3.sel(mode = mode)[::step, ::step], data4.sel(mode = mode)[::step, ::step],
                            scale=10, width=0.002, color='k', zorder=2)
        ax.contour(data5.longitude, data5.latitude, data5,colors = 'r',levels = [0])
        ax.plot(data6.longitude,data6,'b')
        
        ax.set_xticks(x_ticks)
        ax.set_xticklabels(x_labels)
        ax.set_xlabel("Longitude")
        ax.set_ylim([y_ticks[0],y_ticks[-1]])
        ax.coastlines(linewidth=0.5, zorder=1,color='gray')
        ax.set_title(f"Mode {mode}  {(100*expvar.sel(mode=mode)):.2f}%")
        ax.tick_params(direction='in', which='both')
        
    axes[0].set_yticks(y_ticks)
    axes[0].set_yticklabels(y_labels)
    axes[0].set_ylabel("Latitude")
    plt.subplots_adjust(wspace=0.05, hspace=0.1)
    # Create color bars
    cbar = plt.colorbar(shadings,ax=axes,
                                 extendrect = True,
                                 extendfrac = 'auto',
                                 orientation='vertical',
                                 #ticks=lb_ticks,
                                 shrink=0.45,
                                 aspect = 20,
                                 drawedges=False,
                                 pad=0.02)
    cbar.ax.tick_params(labelsize=13)
    #cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_label(label=title, size=13,loc = 'center')
    plt.savefig(fout,bbox_inches='tight',pad_inches = 0.07)
    return

if __name__ == "__main__":
    main()