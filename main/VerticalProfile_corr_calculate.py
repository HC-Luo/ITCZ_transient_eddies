#calculate vertival profiles of correlation coefficient of v and MSE on the two sides of EFE
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
    fpath2 = "/glade/derecho/scratch/hcluo/"
    fpath3 = "/glade/work/hcluo/data/MSE_adjust/"
    fpath_out = "/glade/derecho/scratch/hcluo/VerticalProfile/5NS/"
    #begin = int(sys.argv[1])
    #years = [begin]
    years = range(1997,2024)
    #fout1 = fpath_out+"v_MSE_corr_"+f"{years[0]}.nc"
    fout1 = fpath_out+"v_MSE_corr.nc"

    months = [1,2,3,4,5,6,7,8,9,10,11,12]
    MONTHS = ["%02d" % x for x in np.arange(1,13)]
    days = ["%02d" % x for x in np.arange(1,32)]
    Cp = 1004.0
    Lv = 2.5e6
    BOUNDARY = [-5,5,0,360]
    spatial_resolution = 0.5 #degree
    temporal_resolution = 12 #hrs, can only be 1, 2, 3, 4, 6, 12
    a = 6.371e6  
    sum_x = None
    sum_y = None
    n_total = 0
    
    for i in years:
        fnames0 = []
        fnames1 = fpath2+"MSE/MSE_anno_"+str(i)+".nc"
        fnames2 = fpath2+"v_adj/v_adj_anno_"+str(i)+".nc"

        for j in np.arange(1,13):
            year_month = str(i)+"%02d" % j
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

        ds1 = read_data(fnames1,BOUNDARY,temporal_resolution,spatial_resolution)
        MSE_anno = xr.decode_cf(ds1,decode_times = True)['MSE_anno']
        del ds1
        gc.collect()
    
        ds6 = read_data(fnames2,BOUNDARY,temporal_resolution,spatial_resolution)
        v_adj_anno = xr.decode_cf(ds6,decode_times = True)['v_adj_anno']
        del ds6
        gc.collect()
    
        ds8 = read_data2(fnames0,BOUNDARY,temporal_resolution,spatial_resolution)/100.0
        sp = xr.decode_cf(ds8,decode_times = True)["SP"]
        mask = maskout(MSE_anno,sp)#xr.ones_like(vMSE)#

        v_adj_anno = xr.where(mask==1, v_adj_anno,np.nan)
        MSE_anno = xr.where(mask==1, MSE_anno,np.nan)

        sx = v_adj_anno.sum(dim="time")
        sy = MSE_anno.sum(dim="time")
        n  = MSE_anno.sizes["time"]
    
        sum_x  = sx if sum_x is None else sum_x + sx
        sum_y  = sy if sum_y is None else sum_y + sy
        n_total += n
    
    mean_x = sum_x / n_total   # shape: (lat, lon)
    mean_y = sum_y / n_total
    
    # ── Pass 2: accumulate covariance / variance terms ────────────────────
    cov_xy = None
    var_x  = None
    var_y  = None
    
    for i in years:
        fnames0 = []
        fnames1 = fpath2+"MSE/MSE_anno_"+str(i)+".nc"
        fnames2 = fpath2+"v_adj/v_adj_anno_"+str(i)+".nc"

        for j in np.arange(1,13):
            year_month = str(i)+"%02d" % j
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

        ds1 = read_data(fnames1,BOUNDARY,temporal_resolution,spatial_resolution)
        MSE_anno = xr.decode_cf(ds1,decode_times = True)['MSE_anno']
        del ds1
        gc.collect()
    
        ds6 = read_data(fnames2,BOUNDARY,temporal_resolution,spatial_resolution)
        v_adj_anno = xr.decode_cf(ds6,decode_times = True)['v_adj_anno']
        del ds6
        gc.collect()
    
        ds8 = read_data2(fnames0,BOUNDARY,temporal_resolution,spatial_resolution)/100.0
        sp = xr.decode_cf(ds8,decode_times = True)["SP"]
        mask = maskout(MSE_anno,sp)#xr.ones_like(vMSE)#

        v_adj_anno = xr.where(mask==1, v_adj_anno,np.nan)
        MSE_anno = xr.where(mask==1, MSE_anno,np.nan)
    
        dx = v_adj_anno - mean_x   # broadcasting: (time,lat,lon) - (lat,lon)
        dy = MSE_anno - mean_y
    
        c  = (dx * dy).sum(dim="time")
        vx = (dx ** 2).sum(dim="time")
        vy = (dy ** 2).sum(dim="time")
    
        cov_xy = c  if cov_xy is None else cov_xy + c
        var_x  = vx if var_x  is None else var_x  + vx
        var_y  = vy if var_y  is None else var_y  + vy
    
    # ── Final correlation map ─────────────────────────────────────────────
    v_MSE_corr = cov_xy / np.sqrt(var_x * var_y)  
    del v_adj_anno,MSE_anno
    gc.collect()

    v_MSE_corr_mean = v_MSE_corr.mean(["latitude","longitude"],skipna=True)
    del v_MSE_corr
    gc.collect()

    ds_out1 = xr.Dataset(data_vars = {"v_MSE_corr_vertical":("level",v_MSE_corr_mean.data)},
                        coords = {"level":v_MSE_corr_mean.level},
                        attrs = {"description":str(temporal_resolution)+" hourly vertical profile of eddy v and MSE correlation coefficient",
                                 "period": str(years[0])+" to "+str(years[-1])})
    ds_out1.to_netcdf(path = fout1) 

    logging.info("finished writting")