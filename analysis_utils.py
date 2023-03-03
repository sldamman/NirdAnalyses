import numpy as np, xarray as xr, wrf, netCDF4 as nc
import functools, operator

def calc_density(data):
    P = data['P'] + data['PB']
    theta = data['T'] + 300 #+ data['T00'] NO clue why
    T = wrf.tk(P, theta)
    Tv = wrf.tvirtual(T, data['QVAPOR'], units='K')
    rho = P / (287 * Tv)
    return rho

def convert_to_gm3(var, rho):
    ''' Method for converting mizing ratios of units kg/kg to g/m3. 
        This requires a lot of memory so it cannot be done for all levels simultanously. '''
    return var * 1000 * rho

def convert_to_m3(var, rho):
    ''' Method for converting mizing ratios of units kg/kg to g/m3. 
        This requires a lot of memory so it cannot be done for all levels simultanously. '''
    return var * rho

############################################### feb 10
def all_ice(data, t=None, mass=False):
    if mass:
        vars = ['QICE', 'QGRAUP', 'QSNOW']
    else:
        vars = ['QNICE', 'QNSNOW', 'QNGRAUPEL']

    if t is not None:
        ice = data[vars[0]].isel(Time=t).where(data[vars[0]].isel(Time=t) > 0, other=0)
        for e in vars[1:]:
            ice += data[e].isel(Time=t).where(data[e].isel(Time=t) > 0, other=0)
        return ice

    else:
        ice = data[vars[0]].where(data[vars[0]] > 0, other=0)
        for e in vars[1:]:
            ice += data[e].where(data[e] > 0, other=0)
        return ice

def all_liquid(data, t=None, mass=False):
    if mass:
        vars = ['QCLOUD', 'QRAIN']
    else:
        vars = ['QNCLOUD', 'QNRAIN']

    if t is not None:
        liquid = data[vars[0]].isel(Time=t).where(data[vars[0]].isel(Time=t) > 0, other=0) + \
        data[vars[1]].isel(Time=t).where(data[vars[1]].isel(Time=t) > 0, other=0)
        return liquid

    else:
        liquid = data[vars[0]].where(data[vars[0]] > 0, other=0) + \
        data[vars[1]].where(data[vars[1]] > 0, other=0)
        return liquid

def cldmsk(inp, data, t=None, cld_thrsh=0.50):
    if t is not None:
        return inp * xr.where(data.CLDFRA.isel(Time=t) > cld_thrsh, x=1, y=np.nan)
    else:
        return inp * xr.where(data.CLDFRA > cld_thrsh, x=1, y=np.nan)

def clip_to_region(data, datapath, minll=(78, 11), maxll=(79, 12), clip_to_data=None):#minlat, minlon, maxlat, maxlon):
    if clip_to_data is not None:
        Ds = nc.Dataset(clip_to_data)
        mxx, mxy = Ds['XLONG'].shape[2] - 1, Ds['XLONG'].shape[1] - 1
        minll = wrf.xy_to_ll(Ds, 0, 0)
        maxll = wrf.xy_to_ll(Ds, mxy, mxx)
        
    minx, miny = [int(e) for e in wrf.ll_to_xy(nc.Dataset(datapath), float(minll[0]), float(minll[1]))]
    maxx, maxy = [int(e) for e in wrf.ll_to_xy(nc.Dataset(datapath), float(maxll[0]), float(maxll[1]))]
    return data.isel(west_east=slice(minx, maxx), south_north=slice(miny, maxy))

def lndmsk(data):
    return xr.where(data.XLAND.isel(Time=1) == 1, x=1, y=np.nan)

def wtrmsk(data):
    return xr.where(data.XLAND.isel(Time=1) == 2, x=1, y=np.nan)
    
def iwf_calc(data, mass=False, t=None, cld_thrsh=0.5, clip_to_cloud=True):
    ALL_ICE = all_ice(data, mass=mass, t=t)
    ALL_LIQUID = all_liquid(data, mass=mass, t=t)
    return cldmsk(ALL_ICE / (ALL_ICE + ALL_LIQUID), data=data, cld_thrsh=cld_thrsh, t=t)

def iwf_to_mask(iwf, thrsh=0.1):
    iwf_masked = xr.where(iwf > 1 - thrsh, x = 1, y = iwf)
    iwf_masked = xr.where(iwf <= thrsh, x = 0, y = iwf)
    iwf_masked = xr.where((iwf > thrsh) & (iwf < 1 - thrsh), x = 0.5, y = iwf)
    return iwf_masked

def composition_msk(data, thrsh=0.05, iwf=None, mass=True):
    if iwf is None:
        iwf = iwf_calc(data, mass=mass)

    mixed = iwf.where(iwf > thrsh, other=np.nan)
    mixed = xr.where(mixed < 1 - thrsh, x = 1, y=np.nan)

    ice = xr.where(iwf > 1 - thrsh, x=1, y=np.nan)
    liquid = xr.where(iwf < thrsh, x=1, y=np.nan)
    return liquid, mixed, ice

def find_top(data, datapath, cld_thrsh=0.5, t=None):
    h = wrf.getvar(nc.Dataset(datapath), 'z').to_numpy()

    if t is not None:
        hc = xr.where(data.CLDFRA.isel(Time=t, bottom_top=slice(0,84)) > cld_thrsh, x=1, y=0) * h[:84, :, :]
    else:
        hc = xr.where(data.CLDFRA.isel(bottom_top=slice(0,84)) > cld_thrsh, x=1, y=0) * h[:84, :, :]
    
    cloudtop = hc.argmax(dim='bottom_top')
    return cloudtop.rename('Cloud-top level')

def select_top(indata, cloudtop, n_levels=10):
    return indata.where((indata.bottom_top > cloudtop - n_levels) 
                        & (indata.bottom_top <= cloudtop) 
                        & (indata.bottom_top != 0), other=np.nan)

def calc_T(data, t):
    return wrf.tk(data.P.isel(Time=t) + data.PB.isel(Time=t), data.T.isel(Time=t) + 300) - 273.15
def calc_W(data, t):
    return wrf.destagger(data.W.isel(Time=t), stagger_dim=1, meta=True)

def consecutive(arr):
    return np.split(arr, np.where(np.diff(arr) != 1)[0] + 1)

import time
class timer:
    def __init__(self):
        self._start_time = None
        self._round_time = None

    def start(self):
        self._start_time = time.perf_counter()

    def round(self):
        if self._round_time is None:
            t = time.perf_counter() - self._start_time
            self._round_time = time.perf_counter()
        else:
            t = time.perf_counter() - self._round_time
            self._round_time = time.perf_counter()
        return f'Elapsed time: {t:0.4f} seconds'
    
    def stop(self):
        t = time.perf_counter() - self._start_time
        self._start_time = None
        self._round_time = None
        return f'Elapsed time: {t:0.4f} seconds.'
        
def flatten_list(lst):
    try:
        return functools.reduce(operator.iconcat, lst, [])
    except TypeError:
        return []