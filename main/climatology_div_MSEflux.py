#Composites of divergent component of MSE flux
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


def map_plot(Data,Data2,cmap,fileout):
    data = Data[0]
    west_bd = min(data.longitude)
    east_bd = max(data.longitude)
    south_bd = min(data.latitude)
    north_bd = max(data.latitude)


    ctitles = r"$\overline{\langle v h \rangle_D}\;(\times 10^8 \, W\,m^{-1})$"
    
    titles2 = r"$\overline{[\langle v h \rangle_D]}\;(\times 10^8 \, W\,m^{-1})$"
    
    titles3 = r"$\overline{[P]}\;(mm\,day^{-1})$"

    # Specify contour levels and contour ticks
    cn_levels = np.linspace(-2, 2,9)
    cn_levels2 = np.arange(4,40,4)
    lb_ticks  = cn_levels

    x_ticks  = np.linspace(-2.0,2.0,5)
    x_ticks2 = np.linspace(0,9,4)
    y_ticks = np.arange(south_bd, north_bd+10, 10)

    projection = ccrs.PlateCarree(central_longitude=180.0)

    fig = plt.figure(figsize=(15, 10),layout='compressed',constrained_layout=True)#,
                      #gridspec_kw=dict(hspace=0.03,wspace = 0.04),
                      
    
    gs = fig.add_gridspec(5, 2,width_ratios=(3.5, 1),wspace=0.0, hspace=0.0)

    ax1 = fig.add_subplot(gs[0,0],projection = projection)
    ax2 = fig.add_subplot(gs[1,0],projection = projection)
    ax3 = fig.add_subplot(gs[2,0],projection = projection)
    ax4 = fig.add_subplot(gs[3,0],projection = projection)
    ax5 = fig.add_subplot(gs[4,0],projection = projection)

    bx1 = fig.add_subplot(gs[0,1])
    bx2 = fig.add_subplot(gs[1,1])
    bx3 = fig.add_subplot(gs[2,1])
    bx4 = fig.add_subplot(gs[3,1])
    bx5 = fig.add_subplot(gs[4,1])
    ax = [ax1,ax2,ax3,ax4,ax5]
    bx = [bx1,bx2,bx3,bx4,bx5]
        
    # Create the Axes.
    for i in range(5):
        ax[i].coastlines(linewidth=0.5, zorder=1,color='gray')
        # Use geocat.viz.util convenience function to set axes tick values


        gv.set_axes_limits_and_ticks(ax[i],
                                     #xlim=[west_bd-180, east_bd-180],
                                     ylim=[south_bd, north_bd],
                                     xticks=np.arange(west_bd-180, east_bd-180+30, 30),
                                     yticks=np.arange(south_bd, north_bd+10, 10))
        # Use geocat.viz.util convenience function to add minor and major tick lines
        gv.add_major_minor_ticks(ax[i], x_minor_per_major=4,
                                      y_minor_per_major=2,
                                      labelsize=13)
        # Use geocat.viz.util convenience function to make plots look like NCL plots by
        # using latitude, longitude tick labels
        gv.add_lat_lon_ticklabels(ax[i])
        gv.set_titles_and_labels(ax[i],ylabel="Latitude",labelfontsize=13)
    
        # Remove the degree symbol from tick labels
        ax[i].xaxis.set_major_formatter(LongitudeFormatter(degree_symbol=''))
        ax[i].yaxis.set_major_formatter(LatitudeFormatter(degree_symbol=''))
        ax[i].tick_params(which='both', direction='in', labelsize=13)
        # Use geocat.viz.util convenience function to set titles and labels
        #gv.set_titles_and_labels(ax[i], maintitle = titles[i], maintitlefontsize=13)
    
        shadings = ax[i].contourf(Data[i].longitude-180,Data[i].latitude,Data[i],
        levels=cn_levels,
        cmap=cmap,zorder=0,extend='both')

        #ax[i].contour(Data[i].longitude-180,Data[i].latitude,Data[i].rolling(longitude = 20, min_periods = 1,center=True).mean(),
        #levels=0,
        #zorder=1,colors="k",linewidths = 0.8)

        contour = ax[i].contour(Data2[i].longitude-180,Data2[i].latitude,Data2[i],
        levels=cn_levels2,
        zorder=1,colors="g",linewidths = 0.8)
        if i == 1:
            ax[i].contour(Data2[i].longitude-180,Data2[i].latitude,Data2[i],
            levels=0,
            zorder=2,colors="g",linewidths = 1.1)
        #else:
        #    ax[i].plot(Data2[i].longitude-180,Data2[i].idxmax(dim="latitude").rolling(longitude=5, min_periods = 1,center=True).mean(),"k",linewidth=1.0,zorder = 2)
#
        #ax[i].clabel(contour, levels = None, inline=True, fontsize=8)


        Data_zonal_mean = Data[i].mean(dim = 'longitude')
        Data_zonal_mean2 = Data2[i].mean(dim = 'longitude')
        bx[i].plot(Data_zonal_mean,Data_zonal_mean.latitude,"r",
                     np.zeros(len(Data_zonal_mean.latitude)),Data_zonal_mean.latitude,"k:",
                     Data_zonal_mean,np.zeros(len(Data_zonal_mean)),"k:")
        
        bx[i].set_xticks(x_ticks)
        bx[i].set_yticks(y_ticks)
        bx[i].set_xlim(x_ticks[0],x_ticks[-1])
        bx[i].set_ylim(y_ticks[0],y_ticks[-1])
        bx[i].tick_params(which="both",direction="in", labelleft=False)
        bx[i].tick_params(axis='x', labelcolor='r')
        bx[i].grid(linestyle = ':', linewidth = 0.5)

        cx = bx[i].twiny()
        cx.plot(Data_zonal_mean2,Data_zonal_mean2.latitude,"g")
        cx.set_xticks(x_ticks2)
        cx.set_xlim(x_ticks2[0],x_ticks2[-1])
        cx.tick_params(axis='x', labelcolor='g')
        cx.tick_params(which="both",direction="in", labelleft=False)
        if i == 0:
            cx.set_xlabel(titles3,color = "g",fontsize=13)

    # Create color bars
    cbar = plt.colorbar(shadings,ax=ax[-1],
                                 extendrect = True,
                                 extendfrac = 'auto',
                                 orientation='horizontal',
                                 ticks=lb_ticks,
                                 shrink=0.98,
                                 aspect = 50,
                                 drawedges=False)#,
                                 #pad=0.04)
    cbar.ax.tick_params(labelsize=13)
    cbar.ax.xaxis.set_label_position('bottom')
    cbar.set_label(label=ctitles, size=13,loc = 'center')
    gv.set_titles_and_labels(ax[-1], xlabel="Longitude",labelfontsize=13)

    bx[-1].set_xlabel(titles2,color = "r",fontsize=13)    

    plt.savefig(fileout,bbox_inches='tight',pad_inches = 0.05)
    return



if __name__ == "__main__":
    
    fpath1 = "/glade/campaign/univ/uccn0006/"
    fpath3 = "/glade/campaign/collections/rda/data/d728007/gpcp_v1.3_daily/"

    years = np.arange(1997,2024)
    months = [[1,2,12],[3,4,5],[6,7,8],[9,10,11],[1,2,3,4,5,6,7,8,9,10,11,12]]
    MONTHS = ["%02d" % x for x in np.arange(1,13)]
    days = ["%02d" % x for x in np.arange(1,32)]
    a = 6371.0*1e3
    BOUNDARY = [-30,30,0,360]
    spatial_resolution = 0.5 #degree
    spatial_resolution2 = 1
    temporal_resolution = 12 #hrs, can only be 1, 2, 3, 4, 6, 12
    temporal_resolution2 = 1 #days
    lag = 0 #days
    fnames1 = []
    fnames3 = []
    for i in years:
        fname1 = "vMSE_col/div_vMSE_col_"+f"{i}.nc"
        fnames1.append(fpath1+fname1)

        for j in np.arange(1,13):
            year_month = str(i)+MONTHS[j-1]

            for k in days:
                fname3 = "gpcp_v01r03_daily_d"+year_month+k+".nc"
                folder3 = os.listdir(fpath3+str(i)+"/")
                if np.isin(fname3,folder3):
                    fnames3.append(fpath3+str(i)+"/"+fname3)

    mask = read_data("/glade/work/hcluo/pro1/v2/landmask.nc",BOUNDARY,temporal_resolution,spatial_resolution)["mask"][0,...]
    ds1 = read_data(fnames1,BOUNDARY,temporal_resolution,spatial_resolution)
    ds1['time'] = ds1['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    Data1 = xr.decode_cf(ds1,decode_times = True)['div_vMSE_col']


    ds3 = read_data(fnames3,BOUNDARY,temporal_resolution2,spatial_resolution2)
    Data3 = xr.decode_cf(ds3,decode_times = True)['precip']

    #resample
    Data1_resample = Data1.resample(time = '1D').mean()
    del Data1

    #Data3  = xr.DataArray(data3.values, dims=['time','latitude','longitude'], 
    #                        coords=dict(
    #                        latitude=data3.latitude,
    #                        longitude=data3.longitude,
    #                        time=Data1.time))
    

    #Data1 = Data1.rolling(time=int(12*30*24/temporal_resolution), min_periods = int(12*30*24/temporal_resolution),center=True).mean().dropna("time")
    #Data2 = Data2.rolling(time=int(12*30*24/temporal_resolution), min_periods = int(12*30*24/temporal_resolution),center=True).mean().dropna("time")
    #Data3 = Data3.rolling(time=int(12*30*24/temporal_resolution), min_periods = int(12*30*24/temporal_resolution),center=True).mean().dropna("time")
    Data1_rolling = Data1_resample.rolling(time=int(30*1/temporal_resolution2), min_periods = int(30*1/temporal_resolution2),center=True).mean().dropna("time")
    Data3_rolling = Data3.rolling(time=int(30*1/temporal_resolution2), min_periods = int(30*1/temporal_resolution2),center=True).mean().dropna("time")
    del Data1_resample,Data3
    gc.collect()

    Data1_plot = []
    Data3_plot = []

    for j in range(5):
        Data1_plot.append(Data1_rolling.sel(time = Data1_rolling.time.dt.month.isin(months[j])).mean(dim = "time")/1.0e8)
        #Data3_plot.append(xr.where(mask,Data3_rolling.sel(time = Data3_rolling.time.dt.month.isin(months[j])).mean(dim = "time"),np.nan))
        #Data4_plot.append(xr.where(mask,-wf.ddy(Data4_rolling.sel(time = Data4_rolling.time.dt.month.isin(months[j]))).mean(dim = "time")*1.0e5,np.nan))
        Data3_plot.append(Data3_rolling.sel(time = Data3_rolling.time.dt.month.isin(months[j])).mean(dim = "time"))

    cmap = wf.colormap(color="BluWhiRed")
    fileout = "/glade/work/hcluo/v5/climatology_div_MSEflux.pdf"
    map_plot(Data1_plot,Data3_plot,
             cmap,fileout)
    
    #fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)
    #ax.plot(Data1_plot[0].longitude,abs(Data1_plot[0]).mean('latitude'),"deepskyblue",
    #         Data1_plot[1].longitude,abs(Data1_plot[1]).mean('latitude'),"lightgreen",
    #         Data1_plot[2].longitude,abs(Data1_plot[2]).mean('latitude'),"firebrick",
    #         Data1_plot[3].longitude,abs(Data1_plot[3]).mean('latitude'),"darkorange")
    #ax.set_xlabel("Longitude")
    #ax.set_xticks(np.arange(BOUNDARY[2], BOUNDARY[3]+30, 30))
    #ax.set_xticklabels([0,30,60,90,120,150,180,-150,-120,-90,-60,-30,0])
    #ax.set_ylabel(r"Magnitude $(\times 10^8 \, W\,m^{-1})$")
    #plt.legend(["DJF","MAM","JJA","SON"])
    #plt.savefig("/glade/work/hcluo/v5/climatology_div_MSEflux_line.pdf")
#
#
    #fig, ax = plt.subplots(figsize=(10, 3), tight_layout=True)
    #ax.plot(Data3_plot[0].longitude,abs(Data3_plot[0]).mean('latitude'),"deepskyblue",
    #         Data3_plot[1].longitude,abs(Data3_plot[1]).mean('latitude'),"lightgreen",
    #         Data3_plot[2].longitude,abs(Data3_plot[2]).mean('latitude'),"firebrick",
    #         Data3_plot[3].longitude,abs(Data3_plot[3]).mean('latitude'),"darkorange")
    #ax.set_xlabel("Longitude")
    #ax.set_xticks(np.arange(BOUNDARY[2], BOUNDARY[3]+30, 30))
    #ax.set_xticklabels([0,30,60,90,120,150,180,-150,-120,-90,-60,-30,0])
    #ax.set_ylabel(r"Precip $(mm\,day^{-1})$")
    #plt.legend(["DJF","MAM","JJA","SON"])
    #plt.savefig("/glade/work/hcluo/v5/climatology_div_MSEflux_line2.pdf")
