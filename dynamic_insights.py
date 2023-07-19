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

#TODO: pull this from elwood when it works
from regrid import Resolution


from itertools import count
from enum import Enum, auto
from typing import Any
from types import MethodType
from dataclasses import dataclass


from rich import print, traceback; traceback.install(show_locals=True)


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


class Frequency(Enum):
    monthly = auto()
    yearly = auto()
    decadal = auto()

class ThresholdType(Enum):
    greater_than = auto()
    less_than = auto()
    greater_than_or_equal_to = auto()
    less_than_or_equal_to = auto()
    equal_to = auto()
    not_equal_to = auto()

@dataclass
class Threshold:
    value: float
    type: ThresholdType


@dataclass
class Variable:
    data: xr.Dataset|xr.DataArray #TODO: make just use dataarray
    Frequency: Frequency|str
    Resolution: Resolution|str

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



#TODO:
# - decorator for compile-time functions that automatically adds the step to the list of steps, and checks that the identifier is unique
# - Variables should only use xarray.DataArray, not xarray.Dataset

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
        self.env: dict[str, Variable] = {}

        # keep track of all identifiers while the pipeline is being compiled
        self.compiled_ids: set[str] = set()

        # tmp id counter
        self.tmp_id_counter = count(0)

    
    #TODO
    #perhaps this decorator could be on the _do_ functions? 
    # i.e. rename them to remove the _do_ prefix, and then the decorator splits off a compile version from the runtime version
    def compile(self, method, check_unique_id:bool=True):
        """
        Decorator to make a method a compile-time method
        When a compile-time method is called:
        1. the method is added to the pipeline list of steps
        2. the identifier is checked to be unique

        Example Usage:
            ```
            @compile()
            def load(self, identifier:str, data: CMIP6Data|OtherData, model:Model|None=None):
                <implementation of _do_load()>
            ```
        causes at runtime:
            ```
            self.steps.append((self._do_load, (identifier, data, model)))
            self._assert_id_is_unique_and_mark_used(identifier)
            ```
        """
        raise NotImplementedError()
    
    
    def step_i_repr(self, index:int):
        """get the repr for the i'th step in the pipeline"""
        return self.step_repr(self.steps[index])
    
    def step_repr(self, step:tuple[MethodType, tuple[Any, ...]]):
        """get the repr for the given step in the pipeline"""
        func, args = step
        name = func.__name__
        if func.__name__.startswith('_do_'):
            name = name[4:]
        return f'Pipeline.{name}({", ".join([str(arg) for arg in args])})'

    
    def _next_tmp_id(self) -> str:
        """get the next available tmp id (to be used for intermediate results)"""
        while True: 
            tmp_id = f'__tmp_{next(self.tmp_id_counter)}__'
            if tmp_id not in self.env:
                return tmp_id

    
    def _assert_id_is_unique_and_mark_used(self, identifier:str):
        """
        Assert that the given identifier is unique, and add it to the compiled_ids set
        Should be called inside any compile-time functions that will add a variable to the pipeline namespace
        NOTE: always call after appending the step to the pipeline
        """
        if identifier in self.compiled_ids:
            raise ValueError(f'Tried to reuse identifier "{identifier}" on step {len(self.steps)}: {self.step_repr(-1)}. All identifiers must be unique.')
        self.compiled_ids.add(identifier)

    
    
    def bind_value(self, identifier:str, value: Variable):
        """Bind a value to an identifier in the pipeline namespace"""
        assert identifier not in self.env, f'Identifier "{identifier}" already exists in pipeline namespace. All identifiers must be unique.'
        self.last_set_identifier = identifier
        self.env[identifier] = value


    def get_value(self, identifier:str) -> Variable:
        """Get a value from the pipeline namespace"""
        return self.env[identifier]
    

    def get_last_value(self) -> Variable:
        """Get the last value that was set in the pipeline namespace"""
        assert self.last_set_identifier is not None, 'No value has been set yet'
        return self.env[self.last_set_identifier]


    def set_geo_resolution(self, target:Resolution|str):
        """
        Append a set_resolution step to the pipeline. 
        Resolution can either be a fixed Resolution object, or target the resolution of an existing dataset by name.
        """
        self.steps.append((self._do_set_geo_resolution, (target,)))


    def _do_set_geo_resolution(self, resolution:Resolution|str):
        """
        Sets the current target resolution for the pipeline.
        Regridding is only run during operations on datasets, and only if the dataset is not already at the target resolution.
        """
        self.resolution = resolution


    def set_time_resolution(self, target:Frequency|str):
        """
        Append a set_frequency step to the pipeline.
        Frequency can either be a fixed Frequency object, or target the frequency of an existing dataset by name.
        """
        self.steps.append((self._do_set_time_resolution, (target,)))


    def _do_set_time_resolution(self, frequency:Frequency|str):
        """
        Sets the current target frequency for the pipeline.
        Temporal interpolation is only run during operations on datasets, and only if the dataset is not already at the target frequency.
        """
        self.frequency = frequency

    
    def _assert_pipe_has_resolution(self):
        """
        check if the pipeline has already set a target geo and temporal resolution
        Should be called before adding any binary operations to the pipeline
        """
        time_res, geo_res = False, False
        for func, _ in self.steps:
            if func == self._do_set_time_resolution:
                time_res = True
            elif func == self._do_set_geo_resolution:
                geo_res = True
        if not time_res:
            raise ValueError('Pipeline must have a target temporal resolution before appending binary operations')
        if not geo_res:
            raise ValueError('Pipeline must have a target geo resolution before appending binary operations')

    def load(self, identifier:str, data: CMIP6Data|OtherData, model:Model|None=None):
        """Append data load step to the pipeline"""
        self.steps.append((self._do_load, (identifier, data, model)))
        self._assert_id_is_unique_and_mark_used(identifier)


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

        # rename the data variable to match the given identifier
        var = var.rename({data.value: identifier})
        
        # create a variable container. Set the frequency and resolution to itself
        var = Variable(var, identifier, identifier)

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


    def threshold(self, out_identifier:str, in_identifier:str, threshold:Threshold):
        """
        Append a threshold step to the pipeline. 
        e.g. 
            ```
            threshold('result', 'tasmax', Threshold(308.15, ThresholdType.greater_than))
            ``` 
        is equivalent to:
            ```
            result = tasmax > 308.15
            ```
        """
        self.steps.append((self._do_threshold, (out_identifier, in_identifier, threshold)))
        self._assert_id_is_unique_and_mark_used(out_identifier)
        

    def _do_threshold(self, out_identifier:str, in_identifier:str, threshold:Threshold):
        """Perform execution of a threshold step"""

        #first make sure the data matches the specified resolution and frequency
        if self.resolution is not None and self.env[in_identifier].Resolution != self.resolution:
            tmp_id = self._next_tmp_id()
            self._do_geo_regrid(tmp_id, in_identifier, self.resolution)
            in_identifier = tmp_id
        if self.frequency is not None and self.env[in_identifier].Frequency != self.frequency:
            tmp_id = self._next_tmp_id()
            self._do_time_regrid(tmp_id, in_identifier, self.frequency)
            in_identifier = tmp_id
        
        pdb.set_trace()
        ...


    def time_regrid(self, out_identifier:str, in_identifier:str, target_frequency:Frequency|str):
        """Append a time regrid step to the pipeline"""
        self.steps.append((self._do_time_regrid, (out_identifier, in_identifier, target_frequency)))
        self._assert_id_is_unique_and_mark_used(out_identifier)


    def _do_time_regrid(self, out_identifier:str, in_identifier:str, target_frequency:Frequency|str):
        raise NotImplementedError()
    

    def geo_regrid(self, out_identifier:str, in_identifier:str, target_resolution:Resolution|str):
        """Append a geo regrid step to the pipeline"""
        self.steps.append((self._do_geo_regrid, (out_identifier, in_identifier, target_resolution)))
        self._assert_id_is_unique_and_mark_used(out_identifier)
    
    
    def _do_geo_regrid(self, out_identifier:str, in_identifier:str, target_resolution:Resolution|str):
        raise NotImplementedError()


    def multiply(self, out_identifier:str, in_identifier1:str, in_identifier2:str):
        """Append a multiply step to the pipeline"""
        self.steps.append((self._do_multiply, (out_identifier, in_identifier1, in_identifier2)))
        self._assert_id_is_unique_and_mark_used(out_identifier)

    
    def _do_multiply(self, out_identifier:str, in_identifier1:str, in_identifier2:str):
        raise NotImplementedError()
    

    def country_split(self, out_identifier:str, in_identifier:str, countries:list[str]):
        """Append a country split step to the pipeline"""
        self.steps.append((self._do_country_split, (out_identifier, in_identifier, countries)))
        self._assert_id_is_unique_and_mark_used(out_identifier)

    
    def _do_country_split(self, out_identifier:str, in_identifier:str, countries:list[str]):
        raise NotImplementedError()
    

    def sum(self, out_identifier:str, in_identifier:str, dims:list[str]):
        """Append a sum step to the pipeline"""
        self.steps.append((self._do_sum, (out_identifier, in_identifier, dims)))
        self._assert_id_is_unique_and_mark_used(out_identifier)


    def _do_sum(self, out_identifier:str, in_identifier:str, dims:list[str]):
        raise NotImplementedError()


    def save(self, identifier:str, filepath:str):
        """Append a save step to the pipeline"""
        self.steps.append((self._do_save, (identifier, filepath)))


    def _do_save(self, identifier:str, filepath:str):
        """Perform execution of a save step"""
        var = self.get_value(identifier)
        var.data.to_netcdf(filepath)


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
    
    # e.g. target fixed geo/temporal resolutions
    # pipe.set_geo_resolution(Resolution(0.5, 0.5))
    # pipe.set_time_resolution(Frequency.monthly)

    # e.g. target geo/temporal resolution of existing data in pipeline
    pipe.set_geo_resolution('tasmax')
    pipe.set_time_resolution('tasmax')

    # load the data
    pipe.load('pop', OtherData.population)
    pipe.load('tasmax', CMIP6Data.tasmax, Model.CAS_ESM2_0)
    
    # operations on the data to perform the scenario
    pipe.threshold('heat', 'tasmax', Threshold(308.15, ThresholdType.greater_than))
    pipe.multiply('exposure0', 'heat', 'pop')
    pipe.country_split('exposure1', 'exposure0', ['China', 'India', 'United States', 'Canada', 'Mexico'])
    pipe.sum('exposure2', 'exposure1', dims=['lat', 'lon'])
    pipe.save('exposure2', 'exposure.nc')
    #...

    # run the pipeline
    pipe.execute()

    # extract any results
    res = pipe.get_last_value()
    pop = pipe.get_value('pop')
    tasmax = pipe.get_value('tasmax')




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