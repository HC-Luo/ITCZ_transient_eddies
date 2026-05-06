#hovmoller plot for unadjusted data
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
    print(ds)
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
        fname1 = "unadjusted/vMSE_col/vMSE_col_"+str(i)+".nc"
        fname2 = "unadjusted/vMSE_col_MMC/vMSE_col_MMC_"+str(i)+".nc"
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

    #resample
    Data1_resample = Data1.resample(time = '1D').mean()
    Data2_resample = Data2.resample(time = '1D').mean()
    Data3_resample = Data3.resample(time = '1D').mean()
    Data4_resample = Data4.resample(time = '1D').mean()
    Data6_resample = Data6.resample(time = '1D').mean()

    del Data3,Data1,Data2,Data4,Data6

    #Data3  = xr.DataArray(data3.values, dims=['time','latitude','longitude'], 
    #                        coords=dict(
    #                        latitude=data3.latitude,
    #                        longitude=data3.longitude,
    #                        time=Data1.time))
    

    Data1_rolling = Data1_resample.rolling(time=int(30*1/temporal_resolution2), min_periods = int(30*1/temporal_resolution2),center=True).mean().dropna("time")
    Data2_rolling = Data2_resample.rolling(time=int(30*1/temporal_resolution2), min_periods = int(30*1/temporal_resolution2),center=True).mean().dropna("time")
    Data3_rolling = Data3_resample.rolling(time=int(30*1/temporal_resolution2), min_periods = int(30*1/temporal_resolution2),center=True).mean().dropna("time")
    Data4_rolling = Data4_resample.rolling(time=int(30*1/temporal_resolution2), min_periods = int(30*1/temporal_resolution2),center=True).mean().dropna("time")
    Data5_rolling = Data5.rolling(time=int(30*1/temporal_resolution2), min_periods = int(30*1/temporal_resolution2),center=True).mean().dropna("time")
    Data6_rolling = Data6_resample.rolling(time=int(30*1/temporal_resolution2), min_periods = int(30*1/temporal_resolution2),center=True).mean().dropna("time")

    Data_annual1 = Data1_rolling.groupby('time.dayofyear').mean()
    Data_annual2 = Data2_rolling.groupby('time.dayofyear').mean()
    Data_annual3 = Data3_rolling.groupby('time.dayofyear').mean()
    Data_annual4 = Data4_rolling.groupby('time.dayofyear').mean()
    Data_annual5 = Data5_rolling.groupby('time.dayofyear').mean()
    Data_annual6 = Data6_rolling.groupby('time.dayofyear').mean()

    del Data1_rolling,Data2_rolling,Data3_rolling,Data4_rolling,Data5_rolling,Data6_rolling
    gc.collect()

    bandpass_avg = [Data_annual1.mean(dim = "longitude"),Data_annual2.mean(dim = "longitude"),Data_annual3.mean(dim = "longitude"),Data_annual4.mean(dim = "longitude"),Data_annual5.mean(dim = "longitude"),Data_annual6.mean(dim = "longitude")]
    cn_levels  = [np.linspace(-5,5,11),np.linspace(-5,5,11),np.linspace(-1,1,11),np.linspace(-1,1,11),np.linspace(1,10,10),np.linspace(5,50,10)]
    lb_ticks = cn_levels
    cmap1 = wf.colormap(color="BluWhiRed")
    cmap2 = wf.colormap(color="WhGreBlu")
    fileout = "/glade/work/hcluo/v5/hovmoller_unadj.pdf"

    titles = ["full (PW)",
              "MMC (PW)",
              "stationary (PW)",
              "transient (PW)",
              r"$precip (mm\,day^{-1})$",
              r"$\langle q \rangle  (kg\,m^{-2})$"]

    fig = plt.figure(figsize=(8, 9),layout='compressed')
    
    gs = fig.add_gridspec(3, 2,wspace=0.02, hspace=0.05)

    ax1 = fig.add_subplot(gs[0,0])
    ax2 = fig.add_subplot(gs[0,1])
    ax3 = fig.add_subplot(gs[1,0])
    ax4 = fig.add_subplot(gs[1,1])
    ax5 = fig.add_subplot(gs[2,0])
    ax6 = fig.add_subplot(gs[2,1])

    # Generate the x-tick positions based on the start of each month (DOY of each 1st day of the month)
    month_days = [1, 32, 60, 91, 121, 152, 182, 213, 244, 274, 305, 335]  # First day of each month in DOY (for a common year)

    # Corresponding month labels (could use full month names or abbreviated ones)
    month_labels = ['J', 'F', 'M', 'A', 'M', 'J', 'J', 'A', 'S', 'O', 'N', 'D']
   
    i = 0
    for ax in [ax1,ax2,ax3,ax4,ax5,ax6]:
        # Plot of chosen variable averaged over latitude and slightly smoothed
        if ax == ax5:
            cf = ax.contourf( Data_annual5.dayofyear, Data_annual5.latitude,
                         bandpass_avg[i].transpose("latitude","dayofyear"),
                         levels=cn_levels[i],
                         cmap=cmap2, extend='both',zorder = 1)
            # plot centroid line
            P = bandpass_avg[i].sel(latitude = slice(-20,20))
            P_area = P*np.cos(P.latitude/180*np.pi)
            top_plus_bottom = (P_area+P_area.shift(latitude = 1))
            top_plus_bottom[:,0] = 0
            P_cent = (top_plus_bottom/2*spatial_resolution).cumsum("latitude")
            median = 0.5*(top_plus_bottom/2*spatial_resolution).sum("latitude")
            b = abs(P_cent-median).idxmin(dim = "latitude")
            c = b.rolling(dayofyear=15, min_periods = 1,center=True).mean()
            ax.plot(Data_annual5.dayofyear,c,"k",linewidth=1.8,zorder = 2)

        elif ax == ax6:
            cf = ax.contourf( Data_annual1.dayofyear, Data_annual1.latitude,
                         bandpass_avg[i].transpose("latitude","dayofyear"),
                         levels=cn_levels[i],
                         cmap=cmap2, extend='both',zorder = 1)
        else:
            cf = ax.contourf( Data_annual1.dayofyear, Data_annual1.latitude,
                             bandpass_avg[i].transpose("latitude","dayofyear"),
                             levels=cn_levels[i],
                             cmap=cmap1, extend='both',zorder = 1)
            ax.contour(Data_annual1.dayofyear, Data_annual1.latitude,
                       bandpass_avg[i].transpose("latitude","dayofyear"),
                       levels=0,colors="k",linewidths = 2.0,zorder = 2)
            
        # Create color bars
        cbar = plt.colorbar(cf,ax=ax,
                                     extendrect = True,
                                     extendfrac = 'auto',
                                     orientation='vertical',
                                     ticks=lb_ticks[i],
                                     shrink=0.9,
                                     aspect = 30,
                                     drawedges=False,
                                     pad=0.04)
        cbar.ax.tick_params(labelsize=13)
        cbar.ax.xaxis.set_label_position('bottom')
        cbar.set_label(label=titles[i], size=13,loc = 'center')
        
    
        # Make some ticks and tick labels
        ax.set_ylim(BOUNDARY[0],BOUNDARY[1])
        ax.tick_params(axis="both",direction="in")
        # Set the x-ticks to correspond to the 1st of each month (DOY to month)
        ax.set_xticks(month_days)  # Set x ticks to 1st of each month (day of year)
        ax.set_xticklabels(month_labels)  # Set x tick labels to month names
        # Add labels and title
        ax.set_xlabel('Month')
        ax.set_ylabel('Latitude')
        #ax.set_title('Hovmöller Diagram with Monthly Ticks')
        i+=1
    plt.savefig(fileout,bbox_inches='tight',pad_inches = 0.05)
    #plt.show()
    
