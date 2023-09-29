from __future__ import annotations

import numpy as np
import xarray as xr
from cftime import DatetimeNoLeap

from flowcast.regrid import RegridType
from flowcast.pipeline import Variable

from typing import Callable
from typing_extensions import ParamSpec
from enum import Enum

import pdb

from os.path import dirname, abspath, join

root = dirname(abspath(__file__))


class Scenario(str, Enum):
    ssp126 = 'ssp126'
    ssp245 = 'ssp245'
    ssp370 = 'ssp370'
    ssp585 = 'ssp585'

class Realization(str, Enum):
    r1i1p1f1 = 'r1i1p1f1'
    #TODO: other realizations as needed

class Model(str, Enum):
    CAS_ESM2_0 = 'CAS-ESM2-0'
    FGOALS_f3_L = 'FGOALS-f3-L'
    #TODO: other models as needed

# aliases for keeping track of which is which
GeoRegridType = RegridType
TimeRegridType = RegridType



_P = ParamSpec("_P")
def dataloader(func: Callable[_P, Variable]) -> Callable[_P, Callable[[], Variable]]:
    """Decorator for data loaders. Makes them static and converts them to a callable"""
    #TODO: make the type signature be correct on the wrapped function
    @staticmethod
    def wrapper(*args, **kwargs) -> Callable[[], Variable]:
        return lambda: func(*args, **kwargs)
    
    return wrapper


class DataLoader:
    #make creating instances of this class raise an error
    def __new__(cls):
        raise TypeError(f'Cannot instantiate {cls.__name__} class. Class is static only.')


class OtherData(DataLoader):
    @dataloader
    def population(*, scenario:Scenario) -> Variable:
        """Load the population data"""
        all_years: list[xr.Dataset] = []
        ssp = scenario.value[:-2] # remove the last two characters (e.g., 'ssp126' -> 'ssp1')

        for year in [2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]:
            data = xr                                                                           \
                .open_dataset(f'{root}/data/population/{ssp.upper()}/Total/NetCDF/{ssp}_{year}.nc')    \
                .rename({f'{ssp}_{year}': 'population'})                                        \
                .assign_coords(
                    time=DatetimeNoLeap(year, 1, 1),
                    ssp=('ssp', np.array([scenario.value], dtype='object')) #note for population, only the first number is relevant
                )
            all_years.append(data)

        # combine the scenario data into one xarray
        data = xr.concat(all_years, dim='time')

        # return the result as a Variable
        return Variable(data['population'], TimeRegridType.interp_or_mean, GeoRegridType.conserve)

    @dataloader
    def land_cover() -> Variable:
        """Modis Land Cover Data"""
        modis = xr.open_dataset(f'{root}/data/MODIS/land-use-5km.nc')
        modis = modis['LC_Type1']
        modis['time'] = modis.indexes['time'].to_datetimeindex().map(lambda dt: DatetimeNoLeap(dt.year, dt.month, dt.day))
        modis = modis.drop(['crs'])

        return Variable(modis, TimeRegridType.nearest_or_mode, GeoRegridType.nearest_or_mode)

    
class CMIP6Data(DataLoader):
    @staticmethod
    def cmip6_loader(variable: str, model: Model, realization: Realization, scenario: Scenario) -> xr.DataArray:
        """Data loader for the CMIP6 models"""
        if model == Model.CAS_ESM2_0:
            grid = 'gn'
        elif model == Model.FGOALS_f3_L:
            grid = 'gr'
        else:
            raise ValueError(f'Unrecognized model: {model}. Expected one of: {[*Model.__members__.values()]}')
        
        return xr.open_dataset(f'{root}/data/cmip6/{variable}/{variable}_Amon_{model}_{scenario}_{realization}_{grid}_201501-210012.nc')[variable]

    @dataloader
    def tasmax(*, realization: Realization, scenario: Scenario, model:Model) -> Variable:
        """Load the tasmax data"""
        data = CMIP6Data.cmip6_loader('tasmax', model, realization, scenario)        
        return Variable(data, TimeRegridType.max, GeoRegridType.interp_or_mean)
    
    @dataloader
    def tas(*, realization: Realization, scenario: Scenario, model:Model) -> Variable:
        """Load the tas data"""
        data = CMIP6Data.cmip6_loader('tas', model, realization, scenario)        
        return Variable(data, TimeRegridType.interp_or_mean, GeoRegridType.interp_or_mean)
    
    @dataloader
    def pr(*, realization: Realization, scenario: Scenario, model:Model) -> Variable:
        """Load the pr data"""
        data = CMIP6Data.cmip6_loader('pr', model, realization, scenario)        
        return Variable(data, TimeRegridType.interp_or_mean, GeoRegridType.interp_or_mean)

    #TODO: data loaders for other variables as needed



