#calculate Gross Moist Stability
import numpy as np 
import xarray as xr
import myfun as wf
import gc, logging,os,sys
import calendar
import matplotlib.pyplot as plt

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

def read_data2(fname,BOUNDARY,temporal_resolution,spatial_resolution):
    ds = xr.open_mfdataset(fname,concat_dim = 'time',combine='nested',decode_times=False)
    TIME = ds['time']
    builtin_temporal_resolution = (TIME[1]-TIME[0]).values
    builtin_spatial_resolution = (ds.longitude[1]-ds.longitude[0]).values
    if (len(ds.latitude) > 1) & (ds.latitude[0] > ds.latitude[1]):
        ds = ds.reindex(latitude=ds.latitude[::-1])
    ds = ds.sel(time = slice(TIME[0],TIME[-1],int(temporal_resolution/builtin_temporal_resolution)),
                latitude = slice(min(BOUNDARY[0],BOUNDARY[1]),max(BOUNDARY[0],BOUNDARY[1]),int(spatial_resolution/builtin_spatial_resolution)), 
                longitude = slice(min(BOUNDARY[2],BOUNDARY[3]),max(BOUNDARY[2],BOUNDARY[3]),int(spatial_resolution/builtin_spatial_resolution)))

    logging.info("finished reading data")
    return ds#.reindex(latitude=ds.latitude[::-1])

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

def column_integrate(data, mask):
    return wf.nantrapz(xr.where(mask==1, data,np.nan),data.level,dim = "level")/9.8*100.0

def before(year,month):
    if month == 1:
        year = year-1
        month = 12
    else:
        month = month-1
    return str(year)+"%02d" % month

def after(year,month):
    if month == 12:
        year = year+1
        month = 1
    else:
        month = month+1
    return str(year)+"%02d" % month


def last_day_before(year,month):
    if month == 1:
        year = year-1
        month = 12
    else:
        month = month-1
    return str(calendar.monthrange(year, month)[1])

def last_day_after(year,month):
    if month == 12:
        year = year+1
        month = 1
    else:
        month = month+1
    return str(calendar.monthrange(year, month)[1])


def maskout(model,sp):
    if "level" in model.coords:
        level_exp,sp_exp = xr.broadcast(model.level, sp)
        level_exp = level_exp.transpose(*model.dims)
        sp_exp = sp_exp.transpose(*model.dims)
        Data = xr.DataArray(np.ones(np.shape(sp_exp)),dims = sp_exp.dims,coords = sp_exp.coords)
        mask = xr.where((level_exp <= sp_exp), Data,0)
        logging.info("finished masking")
        return mask

    else:
        raise ValueError("stopped masking, the level coord is not here")
    
def running_mean(val,temporal_resolution):
    return val.rolling(time=int(30*24/temporal_resolution), min_periods = 1,center=True).mean()

def find_wind_reversal_level(da: xr.DataArray) -> xr.DataArray:
    """
    Find the level at which the time-mean, zonal-mean meridional wind
    reverses direction (southerly ↔ northerly).

    Parameters
    ----------
    da : xr.DataArray
        4D DataArray with dims (time, level, latitude, longitude).
        Levels should be ordered surface → top (descending pressure).

    Returns
    -------
    xr.DataArray (latitude,)
        The level at which v changes sign, for each latitude.
    """

    # 1. Time mean + zonal mean → (level, latitude)
    v_bar = da.mean(dim="time").mean(dim="longitude")

    # 2. Find where sign changes along the level axis
    sign_change = np.sign(v_bar).diff(dim="level")  # non-zero where sign flips

    # 3. For each latitude, find the first level where the sign flips
    levels = da["level"].values
    lats   = da["latitude"].values
    transition_levels = []

    for i in range(len(lats)):
        col      = sign_change.isel(latitude=i).values
        flip_idx = np.where(col != 0)[0]

        if flip_idx.size == 0:
            transition_levels.append(np.nan)   # no reversal found
        else:
            transition_levels.append(float(levels[flip_idx[0]]))

    return xr.DataArray(
        transition_levels,
        coords={"latitude": lats},
        dims=["latitude"],
        attrs={"long_name": "Level of meridional wind reversal", "units": da["level"].attrs.get("units", "")},
    )
if __name__ == "__main__":
    fpath_in0 = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc/"
    fpath2 = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/"
    fpath3 = "/glade/work/hcluo/data/MSE_adjust/"
    fpath4 = "/glade/derecho/scratch/hcluo/MSE/"
    fpath_out = "/glade/derecho/scratch/hcluo/"
    begin = int(sys.argv[1])
    years = [begin]
    #years = range(1997,1998)

    #fout1 = fpath_out+"GMS/GMS_MMC_"+f"{years[0]}.nc"
    #fout2 = fpath_out+"GMS/GMS_transient_"+f"{years[0]}.nc"
    fout1 = fpath_out+"GMS/GMS_MMC.nc"
    fout2 = fpath_out+"GMS/v_MMC_col"+f"{years[0]}.nc"

    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    MONTHS = ["%02d" % x for x in np.arange(1,13)]
    days = ["%02d" % x for x in np.arange(1,32)]
    Cp = 1004.0
    Lv = 2.5e6
    BOUNDARY = [-30,30,0,360]
    spatial_resolution = 0.5 #degree
    temporal_resolution = 12 #hrs, can only be 1, 2, 3, 4, 6, 12
    a = 6.371e6  
    fnames0 = []
    #fnames1 = []
    fnames2 = []
    fnames3 = []
    fnames4 = []
    fnames5 = []
    fnames6 = []
    fnames7 = []
    fnames7_now = []
    for i in years:
        if i == years[0]:
            #one month ahead
            year_month = str(i-1)+"12"
            fname4 = "adjustment_"+year_month+".nc"
            if np.isin(fname4,os.listdir(fpath3)):
                fnames4.append(fpath3+fname4)
                fname0 = "e5.oper.an.sfc.128_134_sp.ll025sc."\
                    +str(i-1)+"120100_"+str(i-1)+"123123.nc"
                fnames0.append(fpath_in0+year_month+"/"+fname0)
        
                for k in days:
                    fname6 = "e5.oper.an.pl.128_130_t.ll025sc." \
                    +year_month+k+"00_"+year_month+k+"23.nc"
        
                    fname3 = "e5.oper.an.pl.128_132_v.ll025uv." \
                    +year_month+k+"00_"+year_month+k+"23.nc"
        
                    fname5 = "e5.oper.an.pl.128_133_q.ll025sc." \
                    +year_month+k+"00_"+year_month+k+"23.nc"
        
                    fname7 = "e5.oper.an.pl.128_129_z.ll025sc." \
                    +year_month+k+"00_"+year_month+k+"23.nc"
                    folder = os.listdir(fpath2+year_month+"/")
                    if np.isin(fname3,folder):
                        fnames3.append(fpath2+year_month+"/"+fname3)
                        fnames6.append(fpath2+year_month+"/"+fname6)
                        fnames5.append(fpath2+year_month+"/"+fname5)
                        fnames7.append(fpath2+year_month+"/"+fname7)

        #now
        for j in np.arange(1,13):
            year_month = str(i)+"%02d" % j
            fname4 = "adjustment_"+year_month+".nc"
            fnames4.append(fpath3+fname4)

            last_day = str(calendar.monthrange(i, j)[1])
    
            fname0 = "e5.oper.an.sfc.128_134_sp.ll025sc."\
                +year_month+"0100_"+year_month+last_day+"23.nc"
            if i == years[0] & j == 1:
                fnames0.append(fpath_in0+before(i,j)+"/"+"e5.oper.an.sfc.128_134_sp.ll025sc."\
                    +before(i,j)+"0100_"+before(i,j)+last_day_before(i,j)+"23.nc")
            fnames0.append(fpath_in0+year_month+"/"+fname0)
            if i == years[-1] & j == 12:
                fnames0.append(fpath_in0+after(i,j)+"/"+"e5.oper.an.sfc.128_134_sp.ll025sc."\
                    +after(i,j)+"0100_"+after(i,j)+last_day_after(i,j)+"23.nc")

            for k in days:
                fname6 = "e5.oper.an.pl.128_130_t.ll025sc." \
                +year_month+k+"00_"+year_month+k+"23.nc"

                fname3 = "e5.oper.an.pl.128_132_v.ll025uv." \
                +year_month+k+"00_"+year_month+k+"23.nc"

                fname5 = "e5.oper.an.pl.128_133_q.ll025sc." \
                +year_month+k+"00_"+year_month+k+"23.nc"

                fname7 = "e5.oper.an.pl.128_129_z.ll025sc." \
                +year_month+k+"00_"+year_month+k+"23.nc"
                folder = os.listdir(fpath2+year_month+"/")
                if np.isin(fname3,folder):
                    fnames3.append(fpath2+year_month+"/"+fname3)
                    fnames6.append(fpath2+year_month+"/"+fname6)
                    fnames5.append(fpath2+year_month+"/"+fname5)
                    fnames7.append(fpath2+year_month+"/"+fname7)
                    fnames7_now.append(fpath2+year_month+"/"+fname7)

        if i == years[-1]:
            # one month later
            year_month = str(i+1)+"01"
            fname4 = "adjustment_"+year_month+".nc"
            if np.isin(fname4,os.listdir(fpath3)):
                fnames4.append(fpath3+fname4)
                fname0 = "e5.oper.an.sfc.128_134_sp.ll025sc."\
                    +str(i+1)+"010100_"+str(i+1)+"013123.nc"
                fnames0.append(fpath_in0+year_month+"/"+fname0)
            
                for k in days:
                    fname6 = "e5.oper.an.pl.128_130_t.ll025sc." \
                    +year_month+k+"00_"+year_month+k+"23.nc"
        
                    fname3 = "e5.oper.an.pl.128_132_v.ll025uv." \
                    +year_month+k+"00_"+year_month+k+"23.nc"
        
                    fname5 = "e5.oper.an.pl.128_133_q.ll025sc." \
                    +year_month+k+"00_"+year_month+k+"23.nc"
        
                    fname7 = "e5.oper.an.pl.128_129_z.ll025sc." \
                    +year_month+k+"00_"+year_month+k+"23.nc"
                    folder = os.listdir(fpath2+year_month+"/")
                    if np.isin(fname3,folder):
                        fnames3.append(fpath2+year_month+"/"+fname3)
                        fnames6.append(fpath2+year_month+"/"+fname6)
                        fnames5.append(fpath2+year_month+"/"+fname5)
                        fnames7.append(fpath2+year_month+"/"+fname7)
    
    ds5 = read_data(fnames5,BOUNDARY,temporal_resolution,spatial_resolution)
    q = xr.decode_cf(ds5,decode_times = True)['Q']
    MSE = Lv*q 
    del q,ds5
    gc.collect()

    ds6 = read_data(fnames6,BOUNDARY,temporal_resolution,spatial_resolution)
    t = xr.decode_cf(ds6,decode_times = True)['T']
    MSE = MSE + Cp*t
    del t,ds6
    gc.collect()

    ds7 = read_data(fnames7,BOUNDARY,temporal_resolution,spatial_resolution)
    z = xr.decode_cf(ds7,decode_times = True)['Z']
    MSE = MSE + z
    del z,ds7
    gc.collect()

    ds3   = read_data(fnames3,BOUNDARY,temporal_resolution,spatial_resolution) 
    v = xr.decode_cf(ds3,decode_times = True)['V']

    ds4 = read_data(fnames4,BOUNDARY,temporal_resolution,spatial_resolution) 
    ds4['time'] = ds4['time'].assign_attrs(units = "hours since 1900-01-01 00:00:00",calendar = "gregorian")
    vMSE_adjust = xr.decode_cf(ds4,decode_times = True)['vMSE']
     
    v_adj = v-vMSE_adjust/MSE
    del vMSE_adjust,v 
    gc.collect()

    ds8 = read_data2(fnames0,BOUNDARY,temporal_resolution,spatial_resolution)/100.0
    sp = xr.decode_cf(ds8,decode_times = True)["SP"]
    mask = maskout(MSE,sp).sel(time = MSE.time)#xr.ones_like(vMSE)#

    v_adj = xr.where(mask==1, v_adj,np.nan)
    #MSE = xr.where(mask==1, MSE,np.nan)

    #vMSE_full = v_adj*MSE
    v_mean = running_mean(v_adj,temporal_resolution).mean("longitude",skipna=True).broadcast_like(v_adj)
    v_mean_masked = xr.where(mask==1, v_mean,np.nan)
    v_surface_sign = xr.where(
        v_mean_masked.isel({"level": -1}).isnull(),   # if bottom level is masked
        v_mean_masked.ffill("level").isel({"level": -1}),  # fall back to last valid
        v_mean_masked.isel({"level": -1})
    )
    v_surface_sign = xr.where(v_surface_sign >= 0, 1.0, -1.0)  # (time, lat, lon)

    # Retain only levels where v has the SAME sign as the surface flow
    # (this is the lower-tropospheric / equatorward branch)
    lower_branch_mask = (v_mean_masked * v_surface_sign) > 0  # True where v matches surface sign

    v_lower = v_mean_masked.where(lower_branch_mask)

    #MSE_mean = running_mean(MSE,temporal_resolution).mean("longitude",skipna=True).broadcast_like(v_adj)
    #vMSE_MMC = (v_mean*MSE_mean).broadcast_like(vMSE_full)
    #direction_change_level = abs(v_mean).idxmin("level")
    
    #v_mean = xr.where(mask2==1, v_mean,np.nan)
    v_MMC_col = column_integrate(v_lower, mask)

    '''
    v_adj_anno = v_adj-running_mean(v_adj,temporal_resolution)
    v_adj_anno = v_adj_anno-v_adj_anno.mean("longitude",skipna=True)
    MSE_anno = MSE-running_mean(MSE,temporal_resolution)
    MSE_anno = MSE_anno-MSE_anno.mean("longitude",skipna=True)
    vMSE_transient = v_adj_anno*MSE_anno
    del MSE,MSE_anno
    gc.collect()
    '''

    ds7_now = read_data(fnames7_now,BOUNDARY,temporal_resolution,spatial_resolution)
    TIME_mark = xr.decode_cf(ds7_now,decode_times = True)['time']
    del ds7_now
    gc.collect()
    

    #vMSE_full_col = column_integrate(vMSE_full,mask).sel(time = TIME_mark)
    #v_col = column_integrate(v_adj.sel(level = slice(800,1000)),mask.sel(level = slice(800,1000))).sel(time = TIME_mark)
    
    #vMSE_MMC_col = column_integrate(vMSE_MMC,mask).sel(time = TIME_mark).mean("longitude",skipna=True)
    v_MMC_col = v_MMC_col.sel(time = TIME_mark)#.mean("longitude",skipna=True)
    print(v_MMC_col)
    '''
    del vMSE_MMC
    vMSE_transient_col = column_integrate(vMSE_transient,mask).sel(time = TIME_mark).mean("longitude",skipna=True)
    v_anno_col = column_integrate(v_adj_anno.sel(level = slice(800,1000)),mask.sel(level = slice(800,1000))).sel(time = TIME_mark).mean("longitude",skipna=True)
    del vMSE_transient
    gc.collect()
    GMS_full = vMSE_full_col/v_col
    GMS_MMC = vMSE_MMC_col/v_MMC_col
    GMS_transient = vMSE_transient_col/v_anno_col
    #GMS_MMC = GMS_MMC.sel(latitude = slice(0,10))
    #v_MMC_col = v_MMC_col.sel(latitude = slice(0,10))

    plt.figure()
    plt.plot(GMS_full.latitude,GMS_full.mean(["time","longitude"]),"dodgerblue",label = "full")
    plt.legend()
    plt.grid(linestyle = ':', linewidth = 0.5)
    #plt.ylim(-1.0e4,1.0e4)
    plt.savefig("/glade/work/hcluo/v5/GMS_full.pdf",bbox_inches='tight',pad_inches = 0.05)

    plt.figure()
    plt.plot(v_col.latitude,2*a*np.pi*np.cos(v_col.latitude/180*np.pi)*v_col.mean(["time","longitude"])/1.0e6,"dodgerblue",label = "full")
    #plt.plot(GMS_transient.latitude,GMS_transient.mean("time"),"darkorange",label = "transient")
    plt.legend()
    plt.grid(linestyle = ':', linewidth = 0.5)
    plt.savefig("/glade/work/hcluo/v5/GMS_full_mass.pdf",bbox_inches='tight',pad_inches = 0.05)

    plt.figure()
    plt.plot(GMS_MMC.latitude,GMS_MMC.mean("time"),"dodgerblue",label = "MMC")
    #plt.plot(GMS_transient.latitude,GMS_transient.mean("time"),"darkorange",label = "transient")
    plt.legend()
    plt.grid(linestyle = ':', linewidth = 0.5)
    #plt.ylim(-1.0e4,1.0e4)
    plt.savefig("/glade/work/hcluo/v5/GMS_MMC.pdf",bbox_inches='tight',pad_inches = 0.05)

    
    #ds_out1 = xr.Dataset(data_vars = {"GMS_MMC":(("time","latitude","longitude"),GMS_MMC.data)},
    #                    coords = {"time":GMS_MMC.time,
    #                              "latitude":GMS_MMC.latitude,
    #                              "longitude":GMS_MMC.longitude},
    #                    attrs = {"description":str(temporal_resolution)+" hourly gross moist stability (full)",
    #                             "period": str(years[0])+" to "+str(years[-1]),
    #                             "unit":"J kg^-1"})
    #ds_out1.to_netcdf(path = fout1) 
    #del GMS_MMC,ds_out1
    #gc.collect()
    '''
    ds_out2 = xr.Dataset(data_vars = {"v_MMC_col":(("time","latitude","longitude"),v_MMC_col.data)},
                        coords = {"time":v_MMC_col.time,
                                  "latitude":v_MMC_col.latitude,
                                  "longitude":v_MMC_col.longitude})
    ds_out2.to_netcdf(path = fout2) 
    del ds_out2
    gc.collect()
    logging.info("finished writting")

    plt.figure()
    plt.plot(v_MMC_col.latitude,2*a*np.pi*np.cos(v_MMC_col.latitude/180*np.pi)*v_MMC_col.mean(["time","longitude"])/1.0e6,"dodgerblue",label = "MMC")
    #plt.plot(GMS_transient.latitude,GMS_transient.mean("time"),"darkorange",label = "transient")
    plt.legend()
    plt.grid(linestyle = ':', linewidth = 0.5)
    plt.savefig("/glade/work/hcluo/v5/GMS_MMC_mass.pdf",bbox_inches='tight',pad_inches = 0.05)
    