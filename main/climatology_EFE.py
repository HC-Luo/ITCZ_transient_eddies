#Composites of EFE
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
    years = np.arange(1997,2024)
    months = [1,2,11,12]
    fout = f"climatology_EFE_NDJF.pdf"
    MONTHS = ["%02d" % x for x in np.arange(1,13)]
    days = ["%02d" % x for x in np.arange(1,32)]
    BOUNDARY  = [-30,30,0,360]    
    spatial_resolution = 0.5 #degree
    spatial_resolution2 = 1.0 #degree
    temporal_resolution = 12 #hrs, can only be 1, 2, 3, 4, 6, 12
    temporal_resolution2 = 1 #days    
    fpath1 =  "/glade/campaign/collections/rda/data/d633000/e5.oper.fc.sfc.meanflux/"
    fpath2 = "/glade/derecho/scratch/hcluo/"
    fpath3 = "/glade/campaign/collections/rda/data/d728007/gpcp_v1.3_daily/"

    fnames1 = []
    fnames2 = []
    fnames3 = []
    fnames4 = []
    fnames6 = []

    for i in years:
        fname1 = "vMSE_col/div_vMSE_col_"+f"{i}.nc"
        fnames1.append(fpath2+fname1)
        fname4 = "vMSE_col_transient/vMSE_col_transient_"+f"{i}.nc"
        fnames4.append(fpath2+fname4)
        fname6 = "q_col/q_col_"+str(i)+".nc"
        fnames6.append(fpath2+fname6)
        for j in np.arange(1,13):
            year_month = str(i)+MONTHS[j-1]
            
            for k in days:
                fname3 = "gpcp_v01r03_daily_d"+year_month+k+".nc"
                folder3 = os.listdir(fpath3+str(i)+"/")
                if np.isin(fname3,folder3):
                    fnames3.append(fpath3+str(i)+"/"+fname3)
    
    ds1 = read_data(fnames1,BOUNDARY,temporal_resolution,spatial_resolution)
    #ds1["longitude"] = (ds1["longitude"] + 180) % 360 - 180
    ds1 = ds1.sortby('longitude')
    ds1['time'] = ds1['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data2 = xr.decode_cf(ds1,decode_times = True)['div_vMSE_col']

    ds3 = read_data(fnames3,BOUNDARY,temporal_resolution2,spatial_resolution2)
    #ds3["longitude"] = (ds3["longitude"] + 180) % 360 - 180
    ds3 = ds3.sortby('longitude')
    Data3 = xr.decode_cf(ds3,decode_times = True)['precip']
    
    ds4 = read_data(fnames4,BOUNDARY,temporal_resolution,spatial_resolution)
    #ds4["longitude"] = (ds4["longitude"] + 180) % 360 - 180
    ds4 = ds4.sortby('longitude')
    ds4['time'] = ds4['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data4 = xr.decode_cf(ds4,decode_times = True)['vMSE_transient_col']

    ds6 = read_data(fnames6,BOUNDARY,temporal_resolution,spatial_resolution)
    #ds6["longitude"] = (ds6["longitude"] + 180) % 360 - 180
    ds6 = ds6.sortby('longitude')
    ds6['time'] = ds6['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data6 = xr.decode_cf(ds6,decode_times = True)['q_col']

    data1_plot = Data2.sel(time = Data2.time.dt.month.isin(months)).mean("time")
    data2_plot = Data4.sel(time = Data4.time.dt.month.isin(months)).mean("time")
    data3_plot = precip_centroid(Data3.sel(time = Data3.time.dt.month.isin(months))).mean("time")
    data6_plot = Data3.sel(time = Data3.time.dt.month.isin(months)).idxmax(dim = "latitude").mean("time")#precip_centroid(Data6.sel(time = Data6.time.dt.month.isin(months))).mean("time")

    data1_plot = data1_plot.rolling(longitude=5, min_periods = 1,center=True).mean().dropna("longitude")
    data2_plot = data2_plot.rolling(longitude=5, min_periods = 1,center=True).mean().dropna("longitude")
    data3_plot = data3_plot.rolling(longitude=5, min_periods = 1,center=True).mean().dropna("longitude")
    data6_plot = data6_plot.rolling(longitude=5, min_periods = 1,center=True).mean().dropna("longitude")


    x_ticks = np.arange(-180,180+30,30)
    y_ticks = np.arange(BOUNDARY[0],BOUNDARY[1]+10,10)
    x_labels = [f"{abs(lon)}W" if lon < 0 else (f"{lon}" if lon == 0 else f"{lon}E") for lon in ((x_ticks+360)% 360 - 180)]
    y_labels = [f"{abs(lat)}S" if lat < 0 else (f"{lat}" if lat == 0 else f"{lat}N") for lat in y_ticks]
    
    fig, ax = plt.subplots(
        nrows=1, ncols=1,
        subplot_kw={'projection': ccrs.PlateCarree(central_longitude=180)},
        figsize=(10, 6)
    )
   
    ax.contour(data1_plot.longitude-180, data1_plot.latitude, data1_plot,colors = 'k',levels = [0])
    ax.contour(data2_plot.longitude-180, data2_plot.latitude, data2_plot,colors = 'blueviolet',levels = [0])
    ax.plot(data3_plot.longitude-180,data3_plot,'g')
    ax.plot(data6_plot.longitude-180,data6_plot,'g',linestyle = 'dotted')
    
    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_xlabel("Longitude")
    ax.set_ylim([y_ticks[0],y_ticks[-1]])
    ax.coastlines(linewidth=0.5, zorder=1,color='gray')
    ax.tick_params(direction='in', which='both')
        
    ax.set_yticks(y_ticks)
    ax.set_yticklabels(y_labels)
    ax.set_ylabel("Latitude")
    plt.savefig(fout,bbox_inches='tight',pad_inches = 0.07)


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

    if "level" in ds.coords:
        ds = ds.sel(level = slice(100,1000))

    return ds


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

if __name__ == "__main__":
    main()