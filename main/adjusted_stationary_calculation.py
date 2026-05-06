#calculate stationary eddy component of meridional flux
import numpy as np 
import xarray as xr
import myfun as wf
import gc, logging,os,sys
import calendar

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

if __name__ == "__main__":
    fpath_in0 = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.sfc/"
    fpath1 = "/glade/work/hcluo/v5/data/"
    fpath2 = "/glade/campaign/collections/rda/data/d633000/e5.oper.an.pl/"
    fpath3 = "/glade/work/hcluo/data/MSE_adjust/"
    fpath4 = "/glade/derecho/scratch/hcluo/MSE/"
    fpath_out = "/glade/derecho/scratch/hcluo/"
    begin = int(sys.argv[1])
    years = [begin]

    fout1 = fpath_out+"vMSE_col/vMSE_col_"+f"{years[0]}.nc"
    fout2 = fpath_out+"vMSE_col_MMC/vMSE_col_MMC_"+f"{years[0]}.nc"
    fout3 = fpath_out+"vMSE_col_stationary/vMSE_col_stationary_new_"+f"{years[0]}.nc"
    fout4 = fpath_out+"vMSE_col_transient/vMSE_col_transient_"+f"{years[0]}.nc"
    os.system("rm -r "+fout1)
    os.system("rm -r "+fout2)
    os.system("rm -r "+fout3)
    os.system("rm -r "+fout4)

    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    MONTHS = ["%02d" % x for x in np.arange(1,13)]
    days = ["%02d" % x for x in np.arange(1,32)]
    Cp = 1004.0
    Lv = 2.5e6
    BOUNDARY = [-60,60,0,360]
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
    MSE = xr.where(mask==1, MSE,np.nan)
    v_bar = running_mean(v_adj,temporal_resolution)
    MSE_bar = running_mean(MSE,temporal_resolution)

    vMSE_stationary = ((v_bar-v_bar.mean("longitude",skipna=True))*(MSE_bar-MSE_bar.mean("longitude",skipna=True))).mean("longitude",skipna=True).broadcast_like(MSE)
    del v_adj,MSE
    gc.collect()

    ds7_now = read_data(fnames7_now,BOUNDARY,temporal_resolution,spatial_resolution)
    TIME_mark = xr.decode_cf(ds7_now,decode_times = True)['time']
    del ds7_now
    gc.collect()

    vMSE_stationary_col = column_integrate(vMSE_stationary,mask).sel(time = TIME_mark)
    del vMSE_stationary
    gc.collect()

    ds_out3 = xr.Dataset(data_vars = {"vMSE_stationary_col":(("time","latitude","longitude"),vMSE_stationary_col.data)},
                        coords = {"time":vMSE_stationary_col.time,
                                  "latitude":vMSE_stationary_col.latitude,
                                  "longitude":vMSE_stationary_col.longitude},
                        attrs = {"description":str(temporal_resolution)+" hourly column-integrated meridional moist static energy flux (stationary eddy)",
                                 "period": str(years[0])+" to "+str(years[-1]),
                                 "unit":"W m^-1"})
    ds_out3.to_netcdf(path = fout3) 
    del vMSE_stationary_col,ds_out3
    gc.collect()

   
    logging.info("finished writting")