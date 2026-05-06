# Line plots of seasonal migrations of EFE
import numpy as np 
import xarray as xr
import matplotlib as mpl
import matplotlib.pyplot as plt
import myfun as wf
import math, gc, logging,os,copy
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


if __name__ == "__main__":
    fpath1 = "/glade/derecho/scratch/hcluo/"
    fpath2 = "/glade/campaign/collections/rda/data/d728007/gpcp_v1.3_daily/"
    years = np.arange(1997,2024)
    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    MONTHS = ["%02d" % x for x in np.arange(1,13)]
    days = ["%02d" % x for x in np.arange(1,32)]
    a = 6371.0*1e3

    BOUNDARY = [-30,30,0,360]
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
        fname1 = "vMSE_col/vMSE_col_"+str(i)+".nc"
        fname4 = "vMSE_col_transient/vMSE_col_transient_"+str(i)+".nc"
        fname6 = "q_col/q_col_"+str(i)+".nc"
        fnames1.append(fpath1+fname1)
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
    
    ds4 = read_data(fnames4,BOUNDARY,temporal_resolution,spatial_resolution)
    ds4['time'] = ds4['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data4 = xr.decode_cf(ds4,decode_times = True)['vMSE_transient_col']*1e-15
    Data4 = Data4*2*np.pi*a*np.cos(Data4.latitude/180*np.pi)
    
    ds5 = read_data(fnames5,BOUNDARY,temporal_resolution2,2*spatial_resolution)
    Data5 = xr.decode_cf(ds5,decode_times = True)['precip']

    ds6 = read_data(fnames6,BOUNDARY,temporal_resolution,spatial_resolution)
    ds6['time'] = ds6['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data6 = xr.decode_cf(ds6,decode_times = True)['q_col']

    #resample
    Data1_resample = Data1.resample(time = '1D').mean()
    Data4_resample = Data4.resample(time = '1D').mean()
    Data6_resample = Data6.resample(time = '1D').mean()

    del Data1,Data4,Data6
    gc.collect()


    Data1_rolling = Data1_resample.rolling(time=int(30*1/temporal_resolution2), min_periods = int(30*1/temporal_resolution2),center=True).mean().dropna("time")
    Data4_rolling = Data4_resample.rolling(time=int(30*1/temporal_resolution2), min_periods = int(30*1/temporal_resolution2),center=True).mean().dropna("time")
    Data5_rolling = Data5.rolling(time=int(30*1/temporal_resolution2), min_periods = int(30*1/temporal_resolution2),center=True).mean().dropna("time")
    Data6_rolling = Data6_resample.rolling(time=int(30*1/temporal_resolution2), min_periods = int(30*1/temporal_resolution2),center=True).mean().dropna("time")

    Data_annual1 = Data1_rolling.groupby('time.dayofyear').mean()
    Data_annual4 = Data4_rolling.groupby('time.dayofyear').mean()
    Data_annual5 = Data5_rolling.groupby('time.dayofyear').mean()
    Data_annual6 = Data6_rolling.groupby('time.dayofyear').mean()

    del Data1_rolling,Data4_rolling,Data5_rolling,Data6_rolling
    gc.collect()

    CET = Data_annual1.sel(latitude = slice(-5,5)).mean(dim = ["latitude","longitude"])
    precip_max = Data_annual5.sel(latitude = slice(-20,20)).mean("longitude").idxmax(dim = "latitude")
    eddy_EFE = abs(Data_annual4.mean("longitude")).idxmin(dim = "latitude")
    ITCZ_EFE = abs(Data_annual1.mean("longitude")).idxmin(dim = "latitude")
    q_max = Data_annual6.sel(latitude = slice(-20,20)).mean("longitude").idxmax(dim = "latitude")

        
    P = Data_annual5.sel(latitude = slice(-20,20)).mean("longitude")
    P_area = P*np.cos(P.latitude/180*np.pi)
    top_plus_bottom = (P_area+P_area.shift(latitude = 1))
    top_plus_bottom[:,0] = 0
    P_cent = (top_plus_bottom/2*spatial_resolution).cumsum("latitude")
    median = 0.5*(top_plus_bottom/2*spatial_resolution).sum("latitude")
    precip_centroid = abs(P_cent-median).idxmin(dim = "latitude")


    ##smoothing
    #data1 = data1.rolling(dayofyear=30, min_periods = 1,center=True).mean()
    precip_centroid = precip_centroid.rolling(dayofyear=30, min_periods = 1,center=True).mean()
    precip_max = precip_max.rolling(dayofyear=30, min_periods = 1,center=True).mean()
    ITCZ_EFE = ITCZ_EFE.rolling(dayofyear=30, min_periods = 1,center=True).mean()
    eddy_EFE = eddy_EFE.rolling(dayofyear=30, min_periods = 1,center=True).mean()
    q_max = q_max.rolling(dayofyear=30, min_periods = 1,center=True).mean()
    CET = CET.rolling(dayofyear=30, min_periods = 1,center=True).mean()

    DOY = Data_annual1.dayofyear
    # Generate the x-tick positions based on the start of each month (DOY of each 1st day of the month)
    month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]  # First day of each month in DOY (for a common year)

    # Corresponding month labels (could use full month names or abbreviated ones)
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']


    fig = plt.figure(figsize =(7, 4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(DOY,precip_centroid,"b",label = r"$[P]_{centroid}$")
    ax.plot(DOY,precip_max,":b",label = r"$[P]_{max}$")
    ax.plot(DOY,ITCZ_EFE,"r",label = "full EFE")
    ax.plot(DOY,eddy_EFE,":r",label = "transient eddy EFE")
    ax.plot(DOY,q_max,"orange",label = r"$\langle [q] \rangle_{max}$")


    #plt.ylabel(r"${\overline{[\langle v h \rangle]}}\; (PW)$",fontsize=15)
    # Make some ticks and tick labels
    #ax.set_ylim(BOUNDARY[0],BOUNDARY[1])
    ax.tick_params(axis="both",direction="in")
    # Set the x-ticks to correspond to the 1st of each month (DOY to month)
    ax.set_xticks(month_days)  # Set x ticks to 1st of each month (day of year)
    ax.set_xticklabels(month_labels)  # Set x tick labels to month names
    # Add labels and title
    ax.set_xlabel('Month',fontsize=13)
    ax.set_ylabel(r"Latitude $(^{\circ})$",fontsize=13)
    ax.grid(linestyle = ':', linewidth = 0.5,axis='both')
    #ax.set_title('Hovmöller Diagram with Monthly Ticks')
    ax.legend(loc="upper left")

    cx = ax.twinx()
    cx.plot(DOY,-1.0*CET,"k",linewidth=1.8,label = "-CET")
    #cx.set_xticks(x_ticks2)
    cx.set_ylabel(r'$(PW)$',color = "k",fontsize=13)
    cx.tick_params(axis='y', labelcolor='k')
    cx.tick_params(which="both",direction="in", labelleft=False)
    cx.legend(loc="upper right")
    plt.savefig("/glade/work/hcluo/v5/CET.pdf",bbox_inches='tight',pad_inches = 0.05)
    
