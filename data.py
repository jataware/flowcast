from __future__ import annotations

import numpy as np
import xarray as xr
from cftime import DatetimeNoLeap

from spacetime import Frequency, Resolution
from regrid import RegridType

from typing import Callable
from typing_extensions import ParamSpec
from enum import Enum, auto
from dataclasses import dataclass

import pdb




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

# class CMIP6Data(str, Enum):
#     tasmax = 'tasmax'
#     tas = 'tas'
#     pr = 'pr'
#     #TODO: other variables as needed

# class OtherData(str, Enum):
#     population = 'population'
#     land_cover = 'land_cover'

@dataclass
class Variable:
    data: xr.DataArray
    time_regrid_type: RegridType|None
    geo_regrid_type: RegridType|None

# aliases for keeping track of which is which
GeoRegridType = RegridType
TimeRegridType = RegridType

# def get_available_cmip6_data() -> list[tuple[CMIP6Data, Model, Scenario, Realization]]:
#     # parse all cmip6 data filepaths, and make a list of what is available (variable, model, scenario, realization)
#     pdb.set_trace()
#     ...

# For every type of data, indicate what type of regridding operation is appropriate
# GeoRegridType = RegridType
# TimeRegridType = RegridType
# regrid_map:dict[CMIP6Data|OtherData, tuple[GeoRegridType, TimeRegridType]] = {
#     CMIP6Data.tasmax: (GeoRegridType.interp_mean, TimeRegridType.max),
#     CMIP6Data.tas: (GeoRegridType.interp_mean, TimeRegridType.interp_mean),
#     CMIP6Data.pr: (GeoRegridType.interp_mean, TimeRegridType.interp_mean),
#     OtherData.population: (GeoRegridType.conserve, TimeRegridType.interp_mean),
#     OtherData.land_cover: (GeoRegridType.nearest, None),
#     #TODO: other variables as needed
# }



        # use specific data loader depending on the requested data
        
        # if data == OtherData.population:
        #     var = self.get_population_data(self.scenarios)
        # elif data == OtherData.land_cover:
        #     var = self.get_land_use_data()
        # elif isinstance(data, CMIP6Data):
        #     assert model is not None, 'Must specify a model for CMIP6 data'
        #     var = self.load_cmip6_data(data, model)
        # else:
        #     raise ValueError(f'Unrecognized data type: {data}. Expected one of: {CMIP6Data}, {OtherData}')

        # # grab the corresponding geo/temporal regrid types for this data
        # geo_regrid_type, time_regrid_type = regrid_map[data]

        # # adjust the latlon conventions if needed
        # if 'lat' in var.coords and 'lon' in var.coords:
        #     inplace_set_longitude_convention(var, LongitudeConvention.neg180_180)
        #     #TODO: set latitude_convention

        # # extract dataarray from dataset, and wrap in a Variable (indicating current geo/time resolution is itself)
        # var = Variable(var, name, name, time_regrid_type, geo_regrid_type)

        # # save the variable to the pipeline namespace under the given identifier
        # self.bind_value(name, var)


# #TODO: make this handle DataArrays instead of Datasets 
# def load_cmip6_data(self, variable:CMIP6Data, model:Model) -> xr.DataArray:
#     """get an xarray with cmip6 data from the specified model"""

#     all_realizations: list[xr.Dataset] = []
#     for realization in self.realizations:
#         all_scenarios: list[xr.Dataset] = []
#         for scenario in self.scenarios:
    
#             if model == Model.CAS_ESM2_0:
#                 data = self.CAS_ESM2_0_cmip_loader(variable, realization, scenario)
            
#             elif model == Model.FGOALS_f3_L:
#                 data = self.FGOALS_f3_L_cmip_loader(variable, realization, scenario)

#             #TODO: other models as needed
#             else:
#                 raise ValueError(f'Unrecognized model: {model}. Expected one of: {[*Model.__members__.values()]}')
        
#             # add a scenario coordinate
#             data = data.assign_coords(
#                 ssp=('ssp', np.array([scenario.value], dtype='object'))
#             )
#             all_scenarios.append(data)
        
#         # combine the scenario data into one xarray
#         data = xr.concat(all_scenarios, dim='ssp')
#         data = data.assign_coords(
#             realization=('realization', np.array([realization.value], dtype='object'))
#         )
#         all_realizations.append(data)

#     # combine all the data into one xarray with realization as a dimension
#     dataset = xr.concat(all_realizations, dim='realization')

#     return dataset[variable.value]




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
    def population(scenario:Scenario) -> Variable:
        """Load the population data"""
        all_years: list[xr.Dataset] = []
        ssp = scenario.value[:-2] # remove the last two characters (e.g., 'ssp126' -> 'ssp1')

        for year in [2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]:
            data = xr                                                                           \
                .open_dataset(f'data/population/{ssp.upper()}/Total/NetCDF/{ssp}_{year}.nc')    \
                .rename({f'{ssp}_{year}': 'population'})                                        \
                .assign_coords(
                    time=DatetimeNoLeap(year, 1, 1),
                    ssp=('ssp', np.array([scenario.value], dtype='object')) #note for population, only the first number is relevant
                )
            all_years.append(data)

        # combine the scenario data into one xarray
        data = xr.concat(all_years, dim='time')

        # return the result as a Variable
        return Variable(data['population'], TimeRegridType.interp_mean, GeoRegridType.conserve)

    @dataloader
    def land_cover() -> Variable:
        """Modis Land Cover Data"""
        modis = xr.open_dataset('data/MODIS/land-use-5km.nc')
        modis = modis['LC_Type1']
        modis['time'] = modis.indexes['time'].to_datetimeindex().map(lambda dt: DatetimeNoLeap(dt.year, dt.month, dt.day))
        
        #TODO: handling modis over time? for now just take a single frame
        modis = modis.isel(time=0).drop(['time', 'crs'])

        return Variable(modis, None, GeoRegridType.nearest)

    
class CMIP6Data(DataLoader):
    @dataloader
    def tasmax(*, realization: Realization, scenario: Scenario, model:Model) -> Variable:
        """Load the tasmax data"""
        if model == Model.CAS_ESM2_0:
            data = CMIP6Data.CAS_ESM2_0_cmip_loader('tasmax', realization, scenario)
        elif model == Model.FGOALS_f3_L:
            data = CMIP6Data.FGOALS_f3_L_cmip_loader('tasmax', realization, scenario)
        else:
            raise ValueError(f'Unrecognized model: {model}. Expected one of: {[*Model.__members__.values()]}')
        
        return Variable(data, TimeRegridType.max, GeoRegridType.interp_mean)

    @staticmethod
    def CAS_ESM2_0_cmip_loader(variable: str, realization: Realization, scenario: Scenario) -> xr.DataArray:
        """Data loader for the CAS-ESM2-0 model"""
        print(realization, scenario)
        return xr.open_dataset(f'data/cmip6/{variable}/{variable}_Amon_CAS-ESM2-0_{scenario}_{realization}_gn_201501-210012.nc')[variable]

    @staticmethod
    def FGOALS_f3_L_cmip_loader(variable: str, realization: Realization, scenario: Scenario) -> xr.DataArray:
        """Data loader for the FGOALS-f3-L model"""
        return xr.open_dataset(f'data/cmip6/{variable}/{variable}_Amon_FGOALS-f3-L_{scenario}_{realization}_gr_201501-210012.nc')[variable]

  





    #TODO: other models' data loaders as needed
