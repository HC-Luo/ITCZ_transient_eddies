# line plots flux, precip changes with latitude, unadjusted
import numpy as np 
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import myfun as wf
import math, gc, logging,os
import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.mpl.gridliner import LongitudeFormatter, LatitudeFormatter
import geocat.viz as gv

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


def read_data3(fpath, years, months, ds, BOUNDARY, spatial_resolution, temporal_resolution):
    time = ds.time
    fnames = [] 
    MONTHS = ["%02d" % x for x in np.arange(1,13)]
    for i in years:
        for j in months:
            if MONTHS[j-1] == '01':
                fnames.append(fpath+str(i-1)+"12/"+"/e5.oper.fc.sfc.meanflux.235_055_mtpr.ll025sc."\
                +str(i-1)+"121606_"+str(i)+"010106.nc")


            fname1 = fpath+str(i)+MONTHS[j-1]+"/e5.oper.fc.sfc.meanflux.235_055_mtpr.ll025sc."\
            +str(i)+MONTHS[j-1]+"0106_"+str(i)+MONTHS[j-1]+"1606.nc"
                        
            fnames.append(fname1)
    
    
            if MONTHS[j-1] == '12':
                fname2 = fpath+str(i)+"12/e5.oper.fc.sfc.meanflux.235_055_mtpr.ll025sc."\
                +str(i)+"121606_"+str(i+1)+"010106.nc"
            else:
                fname2 = fpath+str(i)+MONTHS[j-1]+"/e5.oper.fc.sfc.meanflux.235_055_mtpr.ll025sc."\
                +str(i)+MONTHS[j-1]+"1606_"+str(i)+MONTHS[j]+"0106.nc"
    
            fnames.append(fname2)

            if MONTHS[j-1] == '12':
                fnames.append(fpath+str(i+1)+"01/"+"/e5.oper.fc.sfc.meanflux.235_055_mtpr.ll025sc."\
                +str(i+1)+"010106_"+str(i+1)+"011606.nc")
    
    ds = xr.open_mfdataset(fnames,concat_dim = 'forecast_initial_time', combine='nested',decode_times=False)
    
    #slicing and regridding
    if (len(ds.latitude) > 1) & (ds.latitude[0] > ds.latitude[1]):
        ds = ds.reindex(latitude=ds.latitude[::-1])
    builtin_spatial_resolution = (ds.longitude[1]-ds.longitude[0]).values

    ds = ds.sel(forecast_hour = slice(temporal_resolution,12,temporal_resolution),
                latitude = slice(min(BOUNDARY[0],BOUNDARY[1]),max(BOUNDARY[0],BOUNDARY[1]),int(spatial_resolution/builtin_spatial_resolution)), 
                longitude = slice(min(BOUNDARY[2],BOUNDARY[3]),max(BOUNDARY[2],BOUNDARY[3]),int(spatial_resolution/builtin_spatial_resolution)))
    
    de = ds.stack(time=("forecast_initial_time","forecast_hour"))
    data = de['MTPR']
    data = data.transpose("time", "latitude", "longitude")

    Data = xr.DataArray(data.values, dims=['time','latitude','longitude'], 
                            coords=dict(
                            latitude=data.latitude,
                            longitude=data.longitude,
                            time=ds.forecast_initial_time.values[0]+temporal_resolution*np.arange(1,1+len(data.time))))
    
    Data = Data.sel(time = time)
    
    logging.info("Finishing reading data")
    return Data

# precipitation centroid
def precip_centroid(Data):
    P = Data.sel(latitude = slice(-20,20))
    spatial_resolution = abs(P.longitude[1] - P.longitude[0])
    P_area = P*np.cos(P.latitude/180*np.pi)
    top_plus_bottom = (P_area+P_area.shift(latitude = 1))
    top_plus_bottom[:,0] = 0
    P_cent = (top_plus_bottom/2*spatial_resolution).cumsum("latitude")
    median = 0.5*(top_plus_bottom/2*spatial_resolution).sum("latitude")
    lat = abs(P_cent-median).idxmin(dim = "latitude")
    return lat

if __name__ == "__main__":
    
    fpath1 = "/glade/derecho/scratch/hcluo/"
    fpath2 = "/glade/campaign/collections/rda/data/d728007/gpcp_v1.3_daily/"
    years = np.arange(1997,2008)
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    MONTHS = ["%02d" % x for x in np.arange(1,13)]
    days = ["%02d" % x for x in np.arange(1,32)]
    a = 6371.0*1e3

    BOUNDARY = [-60,60,0,360]
    spatial_resolution = 0.5 #degree
    temporal_resolution = 12 #hrs, can only be 1, 2, 3, 4, 6, 12
    temporal_resolution2 = 1 #days
    fnames1 = []
    fnames2 = []
    fnames3 = []
    fnames4 = []
    fnames5 = []
    fnames6 = []
    for i in years:
        fname1 = "unadjusted/vMSE_col/vMSE_col_"+str(i)+".nc"
        fname2 = "unadjusted/vMSE_col_MMC/vMSE_col_MMC_test_"+str(i)+".nc"
        fname3 = "unadjusted/vMSE_col_stationary/vMSE_col_stationary_new_"+str(i)+".nc"
        fname4 = "unadjusted/vMSE_col_transient/vMSE_col_transient_"+str(i)+".nc"
        fname6 = "q_col/q_col_"+str(i)+".nc"
        fnames1.append(fpath1+fname1)
        fnames2.append(fpath1+fname2)
        fnames3.append(fpath1+fname3)
        fnames4.append(fpath1+fname4)
        fnames6.append(fpath1+fname6)
        for j in np.arange(1,13):
            year_month = str(i)+MONTHS[j-1]
            for k in days:
                fname5 = "gpcp_v01r03_daily_d"+year_month+k+".nc"
                folder = os.listdir(fpath2+str(i)+"/")
                if np.isin(fname5,folder):
                    fnames5.append(fpath2+str(i)+"/"+fname5)


    ds1 = read_data(fnames1,BOUNDARY,temporal_resolution,spatial_resolution)
    ds1['time'] = ds1['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data1 = xr.decode_cf(ds1,decode_times = True)['vMSE_full_col']*1e-15
    Data1 = Data1*2*np.pi*a*np.cos(Data1.latitude/180*np.pi)
    
    ds2 = read_data(fnames2,BOUNDARY,temporal_resolution,spatial_resolution)
    ds2['time'] = ds2['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data2 = xr.decode_cf(ds2,decode_times = True)['vMSE_MMC_col']*1e-15
    Data2 = Data2*2*np.pi*a*np.cos(Data2.latitude/180*np.pi)

    ds3 = read_data(fnames3,BOUNDARY,temporal_resolution,spatial_resolution)
    ds3['time'] = ds3['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data3 = xr.decode_cf(ds3,decode_times = True)['vMSE_stationary_col']*1e-15
    Data3 = Data3*2*np.pi*a*np.cos(Data3.latitude/180*np.pi)

    ds4 = read_data(fnames4,BOUNDARY,temporal_resolution,spatial_resolution)
    ds4['time'] = ds4['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data4 = xr.decode_cf(ds4,decode_times = True)['vMSE_transient_col']*1e-15
    Data4 = Data4*2*np.pi*a*np.cos(Data4.latitude/180*np.pi)
    
    ds5 = read_data(fnames5,BOUNDARY,temporal_resolution2,2*spatial_resolution)
    Data5 = xr.decode_cf(ds5,decode_times = True)['precip']

    ds6 = read_data(fnames6,BOUNDARY,temporal_resolution,spatial_resolution)
    ds6['time'] = ds6['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data6 = xr.decode_cf(ds6,decode_times = True)['q_col']






    Data1_resample = Data1.resample(time = '1D').mean()
    Data2_resample = Data2.resample(time = '1D').mean()
    Data3_resample = Data3.resample(time = '1D').mean()
    Data4_resample = Data4.resample(time = '1D').mean()
    Data6_resample = Data6.resample(time = '1D').mean()
    Data5 = xr.where(Data5 < 0, np.nan, Data5)
 

    Data1_mean = Data1_resample.mean(["time","longitude"])
    Data2_mean = Data2_resample.mean(["time","longitude"])
    Data3_mean = Data3_resample.mean(["time","longitude"])
    Data4_mean = Data4_resample.mean(["time","longitude"])
    Data5_mean = Data5.mean(["time","longitude"],skipna = True)
    Data6_mean = Data6_resample.mean("time").mean("longitude")
    latitude = Data2_mean.latitude
    data5_lat = precip_centroid(Data5).mean("time").mean("longitude")


    fig, ax1 = plt.subplots(figsize =(7, 4))

    # Plot 3 lines on ax1 (left y-axis)
    ax1.plot(latitude, np.zeros(len(latitude)),color = "lightgrey", linestyle = 'dashed',linewidth=1.2, zorder=0)

    #ax1.plot(latitude, Data1_mean,"k", 
    #         label="original full AET",zorder=1)
    ax1.plot(latitude, Data1_mean,color = "k",
             label="full", zorder=2)
    ax1.plot(latitude, Data2_mean,color = "r", 
             label="MMC", zorder=3)
    ax1.plot(latitude, Data3_mean,color = "k", linestyle = "dotted",
             label="stationary", zorder=3)
    ax1.plot(latitude, Data4_mean,color = "r", linestyle = "dotted",
             label="transient", zorder=3)

    plt.ylabel(r"$(PW)$",fontsize=15)
    plt.xlabel("Latitude",fontsize=15)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(min(latitude),max(latitude))
    plt.grid(linestyle = ':', linewidth = 0.5)
    plt.tick_params(axis="both",direction="in")

    cx = ax1.twinx()
    cx.plot(Data5_mean.latitude,Data5_mean,"b",
            label = r"$\overline{\overline{[P]}}$", zorder=4)

    cx.plot(data5_lat.values, Data5_mean.sel(latitude = data5_lat,method = 'nearest').values,".",color='b', markersize = 10.0, zorder=5)
    cx.tick_params(axis="both",direction="in")
    cx.set_ylabel(r"$(mm\,day^{-1})$",color = "b",fontsize=13)
    #cx.set_xticks(x_ticks2)
    cx.set_ylim(0,6.5)
    # Combine handles and labels from both axes

    lines1 = ax1.get_lines()
    lines2 = cx.get_lines()
    all_lines = list(lines1) + list(lines2)
    labels = [line.get_label() for line in all_lines]
    
    # Add shared legend to ax1
    ax1.legend(all_lines, labels, loc='upper left')
    
    plt.savefig("/glade/work/hcluo/v5/line_unadj.pdf",bbox_inches='tight',pad_inches = 0.05)

    
