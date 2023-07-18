"""
[CMIP6 generalization tasks]
1. ability to dynamically construct the data pipeline for a scenario
2. ability to dynamically/automatically get new data from cmip6 portal
3. including past 80 years in data with ERA5




---------------------------------------------------------------------------



[1. ability to dynamically construct the data pipeline for a scenario]

framework notes:
- functions that take in one or more datasets/etc., and return a result
  - can have prerequisites on the input datasets, e.g. resolutions must match, etc.



[Dataflow pipeline operations]


rescale to target
- other dataset
- specified resolution


arithmetic between datasets
- add
- subtract
- multiply
- divide

arithmetic over single dataset
- threshold

aggregation over single dataset
- sum
- mean
- min
- max

temporal interpolation
- e.g. decadal to yearly

crop data by country





[SCENARIOS]

--------------- heat exposure ----------------
pop = load pop data [decadal]
tasmax = load tasmax data [monthly]

heat = tasmax > 35°C
heat = #TODO: to yearly. currently it is group by year and mean of heat...

pop = interpolate to yearly

pop, heat = crop (pop, heat) to overlapping time period

heat = regrid to pop

exposure = (heat > 0) * pop

result = split by country (exposure, ['US', 'China', 'India', ...])

--------------- crop suitability --------------
...

---------------------------------------------------------------------------



[2. ability to dynamically/automatically get new data from cmip6 portal]
# web scraper for downloading data, then gpt4 for combining into a single dataframe/xarray
#  ---> process for converting to single xarray is a function defined by the llm the gets saved with processed data + original data



---------------------------------------------------------------------------


[3. including past 80 years in data with ERA5]
TBD

"""






from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr
import rioxarray as rxr # needs to be imported?
from rioxarray.exceptions import NoDataInBounds

from enum import Enum
from typing import Any
from types import MethodType


#TODO: pull this from elwood when it works
from regrid import Resolution



import pdb


"""
TASKS:
- have ssp be a dimension in xarray data
- 
"""



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



class CMIP6Data(str, Enum):
    tasmax = 'tasmax'
    tas = 'tas'
    pr = 'pr'
    #TODO: other variables as needed

class OtherData(str, Enum):
    population = 'population'
    MODIS = 'MODIS'


class Frequency(str, Enum):
    monthly = 'monthly'
    yearly = 'yearly'
    decadal = 'decadal'

# Operations
# - load
# - regrid
# - crop
# - rescale
# - add
# - subtract
# - multiply
# - divide
# - threshold
# - sum
# - mean
# - min
# - max
# - interpolate
# - split


class Pipeline:


    def __init__(self, *,
            realizations: Realization|list[Realization],
            scenarios: Scenario|list[Scenario],
            # models: Model|list[Model]
        ):
        
        # static settings for data to be used in pipeline
        self.realizations: list[Realization] = realizations if isinstance(realizations, list) else [realizations]
        self.scenarios: list[Scenario] = scenarios if isinstance(scenarios, list) else [scenarios]

        assert len(self.realizations) > 0, 'Must specify at least one realization'
        assert len(self.scenarios) > 0, 'Must specify at least one scenario'

        
        # dynamic settings for data to be used in pipeline
        self.resolution: Resolution|str|None = None
        self.frequency: Frequency|str|None = None
        #target country split?
        
        # list of steps in the pipeline DAG
        self.steps: list[tuple[MethodType, tuple[Any, ...]]] = []

        # keep track of the most recent result during pipeline execution
        self.last_set_identifier: str|None = None

        # namespace for binding operation results to identifiers 
        self.env: dict[str, Any] = {}

    def bind_value(self, identifier:str, value: Any):
        """Bind a value to an identifier in the pipeline namespace"""
        self.last_set_identifier = identifier
        self.env[identifier] = value

    def get_value(self, identifier:str):
        """Get a value from the pipeline namespace"""
        return self.env[identifier]
    
    def get_last_value(self):
        """Get the last value that was set in the pipeline namespace"""
        assert self.last_set_identifier is not None, 'No value has been set yet'
        return self.env[self.last_set_identifier]

    def set_resolution(self, target:Resolution|str):
        """
        Append a set_resolution step to the pipeline. 
        Resolution can either be a fixed Resolution object, or target the resolution of an existing dataset by name.
        """
        self.steps.append((self._do_set_resolution, (target,)))

    def _do_set_resolution(self, resolution:Resolution|str):
        #TODO:
        # Only run during operations on datasets, and only run if the dataset is not already at the target resolution.
        self.resolution = resolution

    def set_frequency(self, target:Frequency|str):
        """
        Append a set_frequency step to the pipeline.
        Frequency can either be a fixed Frequency object, or target the frequency of an existing dataset by name.
        """
        self.steps.append((self._do_set_frequency, (target,)))

    def _do_set_frequency(self, frequency:Frequency|str):
        self.frequency = frequency

    
    def load(self, identifier:str, data: CMIP6Data|OtherData, model:Model|None=None):
        """Append data load step to the pipeline"""
        self.steps.append((self._do_load, (identifier, data, model)))

    def _do_load(self, identifier:str, data: CMIP6Data|OtherData, model:Model|None=None):
        """Perform execution of a data load step"""

        # use specific data loader depending on the requested data
        match data:
            case OtherData.population:
                var = self.get_population_data(self.scenarios)
            case OtherData.MODIS:
                raise NotImplementedError()
            case CMIP6Data():
                assert model is not None, 'Must specify a model for CMIP6 data'
                var = self.load_cmip6_data(data, model)
            case _:
                raise ValueError(f'Unrecognized data type: {data}. Expected one of: {CMIP6Data}, {OtherData}')

        #rename the data variable to match the given identifier
        var = var.rename({data.value: identifier})
        
        # save the variable to the pipeline namespace under the given identifier
        self.bind_value(identifier, var)

    def load_cmip6_data(self, variable:CMIP6Data, model:Model) -> xr.Dataset:
        """get an xarray with cmip6 data from the specified model"""

        all_realizations: list[xr.Dataset] = []
        for realization in self.realizations:
            all_scenarios: list[xr.Dataset] = []
            for scenario in self.scenarios:
        
                match model:
                    case Model.CAS_ESM2_0:
                        data = self.CAS_ESM2_0_cmip_loader(variable, realization, scenario)
                    
                    case Model.FGOALS_f3_L:
                        data = self.FGOALS_f3_L_cmip_loader(variable, realization, scenario)

                    #TODO: other models as needed
                    case _:
                        raise ValueError(f'Unrecognized model: {model}. Expected one of: {[*Model.__members__.values()]}')
                
                # add a scenario coordinate
                data = data.assign_coords(
                    ssp=('ssp', np.array([scenario.value], dtype='object'))
                )
                all_scenarios.append(data)
            
            # combine the scenario data into one xarray
            data = xr.concat(all_scenarios, dim='ssp')
            all_realizations.append(data)

        # combine all the data into one xarray with realization as a dimension
        dataset = xr.concat(all_realizations, dim='realization')

        return dataset

    @staticmethod
    def CAS_ESM2_0_cmip_loader(variable: CMIP6Data, realization: Realization, scenario: Scenario) -> xr.Dataset:
        """Data loader for the CAS-ESM2-0 model"""
        return xr.open_dataset(f'data/cmip6/{variable}/{variable}_Amon_CAS-ESM2-0_{scenario}_{realization}_gn_201501-210012.nc')

    @staticmethod
    def FGOALS_f3_L_cmip_loader(variable: CMIP6Data, realization: Realization, scenario: Scenario) -> xr.Dataset:
        """Data loader for the FGOALS-f3-L model"""        
        return xr.open_dataset(f'data/cmip6/{variable}/{variable}_Amon_FGOALS-f3-L_{scenario}_{realization}_gr_201501-210012.nc')
        
    @staticmethod
    def get_population_data(scenarios: list[Scenario]) -> xr.Dataset:
        """get an xarray with the specified population data"""

        all_scenarios: list[xr.Dataset] = []
        for scenario in scenarios:

            all_years: list[xr.Dataset] = []
            ssp = scenario.value[:-2] # remove the last two characters (e.g., 'ssp126' -> 'ssp1')

            for year in [2010, 2020, 2030, 2040, 2050, 2060, 2070, 2080, 2090, 2100]:
                data = xr                                                                           \
                    .open_dataset(f'data/population/{ssp.upper()}/Total/NetCDF/{ssp}_{year}.nc')    \
                    .rename({f'{ssp}_{year}': 'population'})                                        \
                    .assign_coords(
                        time=pd.Timestamp(year, 1, 1), 
                        ssp=('ssp', np.array([scenario.value], dtype='object')) #note for population, only the first number is relevant
                    )
                all_years.append(data)

            # combine the scenario data into one xarray
            data = xr.concat(all_years, dim='time')
            all_scenarios.append(data)

        # combine all the data into one xarray with ssp as a dimension
        dataset = xr.concat(all_scenarios, dim='ssp')

        return dataset

    #TODO: other models' data loaders as needed

    
    def execute(self):
        """Execute the pipeline"""
        for func, args in self.steps:
            func(*args)






def get_available_cmip6_data() -> list[tuple[CMIP6Data, Model, Scenario, Realization]]:
    # parse all cmip6 data filepaths, and make a list of what is available (variable, model, scenario, realization)
    pdb.set_trace()
    ...





#TODO: parameterize function with scenario, etc.
def heat_scenario():


    pipe = Pipeline(
        realizations=Realization.r1i1p1f1, 
        scenarios=[Scenario.ssp126, Scenario.ssp245, Scenario.ssp370,Scenario.ssp585], 
    )
    pipe.set_resolution(Resolution(0.5, 0.5))
    pipe.set_frequency(Frequency.monthly)
    pipe.load('pop', OtherData.population)
    pipe.load('tasmax', CMIP6Data.tasmax, Model.CAS_ESM2_0)
    
    # pipe.threshold('heat', 'tasmax', 308.15)
    # pipe.multiply('exposure', 'heat', 'pop')
    # pipe.country_split('exposure', ['China', 'India', 'United States', 'Canada', 'Mexico'])
    # pipe.sum('exposure', 'exposure', dims=['lat', 'lon'])
    # pipe.save('exposure', 'exposure.nc')
    #...

    #TBD on inferring needed transformations, e.g. regridding, etc.

    pipe.execute()

    res = pipe.get_last_value()
    pop = pipe.get_value('pop')
    tasmax = pipe.get_value('tasmax')

    pdb.set_trace()
    ...

    #every step print out: data type and data dims



def crop_scenario():
    raise NotImplementedError()

    pipe = Pipeline(
        realizations=Realization.r1i1p1f1,
        scenarios=[Scenario.ssp126, Scenario.ssp245, Scenario.ssp370,Scenario.ssp585],
    )
    pipe.set_resolution(Resolution(0.5, 0.5))
    pipe.set_frequency(Frequency.monthly)
    pipe.load('tas', CMIP6Data.tas, Model.FGOALS_f3_L)
    pipe.load('pr', CMIP6Data.pr, Model.FGOALS_f3_L)




if __name__ == '__main__':
    heat_scenario()
    # crop_scenario()