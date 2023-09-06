from __future__ import annotations
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
import geopandas as gpd
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask
from cftime import DatetimeNoLeap

#TODO: pull this from elwood when it works
# from elwood.elwood import regrid_dataframe
# from regrid import Resolution
from dataclasses import dataclass
@dataclass
class Resolution:
    dx: float
    dy: float = None

    def __init__(self, dx: float, dy: float|None=None):
        self.dx = dx
        self.dy = dy if dy is not None else dx

import re
from itertools import count
from inspect import signature, getsource, Signature
from enum import Enum, auto
from typing import Any
from types import MethodType
from dataclasses import dataclass


# pretty printing
from rich import print, traceback; traceback.install(show_locals=True)


import pdb


# needed for intellisense to correctly work on @compile decorated methods
def get_signature_from_source(method):
    source = getsource(method)
    match = re.search(r'def\s+\w+\((.*?)\):', source, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Couldn't extract function signature from source")


class LongitudeConvention(Enum):
    ambiguous = auto()
    neg180_180 = auto()
    pos0_360 = auto()

def validate_longitudes(lons:np.ndarray):
    """Validate that the given longitude values are in the range [-180, 360] and are monotonic (either positive or negative)"""
    assert np.all(lons >= -180) and np.all(lons <= 360), f'Longitude values must be in the range [-180, 360]. Got: {lons}'
    deltas = np.diff(lons)
    assert np.all(deltas >= 0) or np.all(deltas <= 0), f'Longitude values must be monotonic (either positive or negative). Got: {lons}'


def validate_latitudes(lats:np.ndarray):
    """Validate that the given latitude values are in the range [-90, 90], and are monotonic (either positive or negative)"""
    assert np.all(lats >= -90) and np.all(lats <= 90), f'Latitude values must be in the range [-90, 90]. Got: {lats}'
    deltas = np.diff(lats)
    assert np.all(deltas >= 0) or np.all(deltas <= 0), f'Latitude values must be monotonic (either positive or negative). Got: {lats}'

def determine_longitude_convention(lons:np.ndarray) -> LongitudeConvention:
    """Determine the longitude convention of the given longitude values"""

    # ensure valid longitude values
    validate_longitudes(lons)
    
    # determine the longitude convention
    if np.all(lons >= 0) and np.all(lons <= 180):
        return LongitudeConvention.ambiguous
    elif np.all(lons >= -180) and np.all(lons <= 180):
        return LongitudeConvention.neg180_180
    elif np.all(lons >= 0) and np.all(lons <= 360):
        return LongitudeConvention.pos0_360
    
    raise ValueError(f'Internal Error: Should be unreachable. Got: {lons}')
    
def convert_longitude_convention(lons:np.ndarray, target_convention:LongitudeConvention) -> np.ndarray:
    """Convert the given longitude values to the specified longitude convention"""

    assert np.all(lons >= -180) and np.all(lons <= 360), f'Longitude values must be in the range [-180, 360]. Got: {lons}'
    
    if target_convention == LongitudeConvention.ambiguous:
        target_convention = LongitudeConvention.neg180_180

    if target_convention == LongitudeConvention.neg180_180:
        return np.where(lons > 180, lons - 360, lons)
    elif target_convention == LongitudeConvention.pos0_360:
        return np.where(lons < 0, lons + 360, lons)
    else:
        raise ValueError(f'Invalid target longitude convention: {target_convention}. Expected one of: {[*LongitudeConvention.__members__.values()]}')


def create_lat_bins(lats:np.ndarray, include_left:bool=True, include_right:bool=True) -> np.ndarray:
    ...

def create_lon_bins(lons:np.ndarray, include_left:bool=True, include_right:bool=True) -> np.ndarray:
    ...



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
    land_cover = 'land_cover'


class Frequency(Enum):
    monthly = auto()
    yearly = auto()
    decadal = auto()

class ThresholdType(Enum):
    greater_than = auto()
    less_than = auto()
    greater_than_or_equal = auto()
    less_than_or_equal = auto()
    equal = auto()
    not_equal = auto()

@dataclass
class Threshold:
    value: float
    type: ThresholdType

class GeoRegridType(Enum):
    sum = auto()
    # mean = auto()
    interp = auto()
    min = auto()
    max = auto()
    nearest = auto()
    none = auto()
    # xr_interp = auto()
    # cdo_sum = auto()
    # cdo_max = auto()
    # cdo_mean = auto()
    # cdo_nearest = auto()
    #TODO: other geo regrid types as needed
    #TODO: pull these from elwood

class TimeRegridType(Enum):
    # mean = auto()
    interp = auto()
    nearest = auto()
    none = auto()
    # xr_interp = auto()
    # xr_nearest = auto()
    #TODO: other temporal regrid types as needed
    #TODO: pull these from elwood


## For every type of data, indicate what type of regridding operation is appropriate
regrid_map:dict[CMIP6Data|OtherData, tuple[GeoRegridType,TimeRegridType]] = {
    CMIP6Data.tasmax: (GeoRegridType.interp, TimeRegridType.interp),
    # CMIP6Data.tas: (GeoRegridType.cdo_mean, TimeRegridType.xr_interp),
    # CMIP6Data.pr: (GeoRegridType.cdo_sum, TimeRegridType.xr_interp),
    OtherData.population: (GeoRegridType.sum, TimeRegridType.interp),
    OtherData.land_cover: (GeoRegridType.nearest, TimeRegridType.none),
    #TODO: other variables as needed
}

@dataclass
class Variable:
    data: xr.DataArray
    frequency: Frequency|str
    resolution: Resolution|str
    time_regrid_type: TimeRegridType
    geo_regrid_type: GeoRegridType

    @staticmethod
    def from_result(data:xr.DataArray, prev:'Variable') -> 'Variable':
        """Create a new Variable from the given data, inheriting the geo/temporal resolution from a previous Variable"""
        return Variable(data, prev.frequency, prev.resolution, prev.time_regrid_type, prev.geo_regrid_type)

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


# Datatypes for representing identifiers in the pipeline namespace
class ResultID(str): ...
class OperandID(str): ...

#TODO:
# - all data loading methods should return xr.DataArray, not xr.Dataset
# - can auto_regrid be handled by the decorator? e.g. by looking at OperandIDs


class Pipeline:


    def __init__(self, *, realizations: Realization|list[Realization], scenarios: Scenario|list[Scenario]):
        
        # static settings for data to be used in pipeline
        self.realizations: list[Realization] = realizations if isinstance(realizations, list) else [realizations]
        self.scenarios: list[Scenario] = scenarios if isinstance(scenarios, list) else [scenarios]

        assert len(self.realizations) > 0, 'Must specify at least one realization'
        assert len(self.scenarios) > 0, 'Must specify at least one scenario'

        # dynamic settings for data to be used in pipeline
        self.resolution: Resolution|str|None = None
        self.frequency: Frequency|str|None = None
        
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

        # bind the current instance to all unwrapped compiled methods (so we don't need to pass the instance manually)
        for attr_name, attr_value in vars(Pipeline).items():
            if hasattr(attr_value, "unwrapped"):
                bound_unwrapped = attr_value.unwrapped.__get__(self, Pipeline)
                setattr(attr_value, "unwrapped", bound_unwrapped)

        # shapefile for country data. Only load if needed
        self._sf = None


    def compile(method:MethodType):
        """
        Decorator to make a method a compile-time method.

        Adds the method to the list of steps in the pipeline (instead of running it)
        checks the the input identifiers already exist in the pipeline namespace
        checks that the result identifier is unique and marks it as used

        NOTE: All identifiers in compiled functions must use either the `ResultID` or `OperandID` type
        identifiers that are not correctly marked will not be properly checked during compilation
        Also all identifiers must be positional arguments, not keyword arguments
        """

        # determine the function signature index of the result and operand identifiers if any
        sig = signature(method)
        params = list(sig.parameters.values())
        result_idx = []
        operand_idx = []
        for i, param in enumerate(params[1:]): #skip self
            if param.annotation == ResultID:
                assert param.kind == param.POSITIONAL_ONLY, f'ERROR compiling "Pipeline.{method.__name__}". Identifier "{param.name}:{param.annotation}" must be a positional-only argument'
                result_idx.append(i)
            elif param.annotation == OperandID:
                assert param.kind == param.POSITIONAL_ONLY, f'ERROR compiling "Pipeline.{method.__name__}". Identifier "{param.name}:{param.annotation}" must be a positional-only argument'
                operand_idx.append(i)


        #TODO: perhaps we don't need this restriction
        assert len(result_idx) <= 1, f'ERROR compiling "Pipeline.{method.__name__}". Compiled functions can only have one result identifier'


        # Check the return annotation
        if sig.return_annotation not in {None, Signature.empty}:
            raise ValueError(f'Method "{method.__name__}" should not have a return annotation or it should be set to None.')


        def wrapper(self:'Pipeline', *args, **kwargs):

            # append the function to the list of steps in the pipeline
            self.steps.append((method, (self,) + args, kwargs))

            # check that any operand identifiers exist in the pipeline namespace
            for i in operand_idx:
                self._assert_id_exists(args[i])
            
            # check that the result identifier is unique and mark it as used
            for i in result_idx:
                self._assert_id_is_unique_and_mark_used(args[i])

        # save the unmodified original function in for use inside pipeline methods
        wrapper.unwrapped = method


        # hacky way to set the wrapper to have the same signature as the original function (for intellisense)
        formatted_sig_args = get_signature_from_source(method)
        formatted_call_args = ', '.join([p.name for p in params])
        wrapper_src = f"""
def intellisense_wrapper({formatted_sig_args}):
    return wrapper({formatted_call_args})
intellisense_wrapper.unwrapped = wrapper.unwrapped
"""

        # Set the namespace for eval
        namespace = {
            'wrapper': wrapper,
            'Pipeline': 'Pipeline',
            **globals(),
        }

        # Execute the source code to create the intellisense friendly wrapper
        exec(wrapper_src, namespace)
        intellisense_wrapper = namespace['intellisense_wrapper']

        return intellisense_wrapper

    
    @staticmethod
    def step_repr(step:tuple[MethodType, tuple[Any, ...], dict[str, Any]]):
        """get the repr for the given step in the pipeline"""
        func, args, kwargs = step
        args = args[1:] #skip self. Removing this causes infinite recursion
        name = func.__name__
        args_str = ', '.join([str(arg) for arg in args])
        kwargs_str = ', '.join([f'{key}={value}' for key, value in kwargs.items()])
        return f'Pipeline.{name}({", ".join((args_str, kwargs_str))})'
    

    def __repr__(self):
        """get the repr for the pipeline"""
        return '\n'.join([f'{i}: {Pipeline.step_repr(step)}' for i, step in enumerate(self.steps)])

    
    def _next_tmp_id(self) -> str:
        """(runtime) get the next available tmp id (to be used for intermediate results)"""
        while True: 
            tmp_id = f'__tmp_{next(self.tmp_id_counter)}__'
            if tmp_id not in self.compiled_ids:
                return tmp_id

    
    def _assert_id_exists(self, identifier:str):
        """
        (Compile-time) assert that the given identifier exists in the pipeline namespace
        Should be called inside any compile-time functions that will use a variable from the pipeline namespace
        NOTE: always call after appending the current step to the pipeline

        @compile handles this automatically
        """
        if identifier not in self.compiled_ids:
            pdb.set_trace()
            raise ValueError(f'Operand identifier "{identifier}" does not exist in pipeline at step {len(self.steps)-1}: {self.step_repr(self.steps[-1])}')
    
    def _assert_id_is_unique_and_mark_used(self, identifier:str):
        """
        (Compile-time) assert that the given identifier is unique, and add it to the compiled_ids set
        Should be called inside any compile-time functions that will add a variable to the pipeline namespace
        NOTE: always call after appending the current step to the pipeline

        @compile handles this automatically
        """
        if identifier in self.compiled_ids:
            raise ValueError(f'Tried to reuse "{identifier}" for result identifier on step {len(self.steps)-1}: {self.step_repr(self.steps[-1])}. All identifiers must be unique.')
        self.compiled_ids.add(identifier)

    
    
    def bind_value(self, identifier:ResultID, value:Variable):
        """Bind a value to an identifier in the pipeline namespace"""
        assert identifier not in self.env, f'Identifier "{identifier}" already exists in pipeline namespace. All identifiers must be unique.'
        self.last_set_identifier = identifier
        self.env[identifier] = value


    def get_value(self, identifier:OperandID) -> Variable:
        """Get a value from the pipeline namespace"""
        return self.env[identifier]
    

    def get_last_value(self) -> Variable:
        """Get the last value that was set in the pipeline namespace"""
        assert self.last_set_identifier is not None, 'No value has been set yet'
        return self.env[self.last_set_identifier]


    @compile
    def set_geo_resolution(self, target:Resolution|str):
        """
        Set the current target geo resolution for the pipeline

        Resolution can either be a fixed Resolution object, e.g. Resolution(0.5, 0.5), 
        or target the resolution of an existing dataset in the pipeline by name, e.g. 'tasmax'

        NOTE: resolution updates are only applied to results of operations, not on data load.
        """
        self.resolution = target


    @compile
    def set_time_resolution(self, target:Frequency|str):
        """
        Set the current target temporal resolution for the pipeline

        Frequency can either be a fixed Frequency object, e.g. Frequency.monthly,
        or target the frequency of an existing dataset in the pipeline by name, e.g. 'tasmax'

        NOTE: frequency updates are only applied to results of operations, not on data load.
        """
        self.frequency = target

    
    def _assert_pipe_has_resolution(self):
        """
        check if the pipeline has already set a target geo and temporal resolution
        Should be called before adding any binary operations to the pipeline
        """
        time_res, geo_res = False, False
        for func, _ in self.steps:
            if func == self.set_time_resolution:
                time_res = True
            elif func == self.set_geo_resolution:
                geo_res = True
        if not time_res:
            raise ValueError('Pipeline must have a target temporal resolution before performing binary operations')
        if not geo_res:
            raise ValueError('Pipeline must have a target geo resolution before performing binary operations')

    
    @compile
    def load(self, name:ResultID, /, data:CMIP6Data|OtherData, model:Model|None=None):
        """
        Load existing data into the pipeline

        Args:
            identifier (str): the identifier to bind the data to in the pipeline namespace
            data (enum): the data to load
            model (enum, optional): the model to load the data from (required for CMIP6 data)
        """

        # use specific data loader depending on the requested data
        
        if data == OtherData.population:
            var = self.get_population_data(self.scenarios)
        elif data == OtherData.land_cover:
            var = self.get_land_use_data()
        elif isinstance(data, CMIP6Data):
            assert model is not None, 'Must specify a model for CMIP6 data'
            var = self.load_cmip6_data(data, model)
        else:
            raise ValueError(f'Unrecognized data type: {data}. Expected one of: {CMIP6Data}, {OtherData}')

        # grab the corresponding geo/temporal regrid types for this data
        geo_regrid_type, time_regrid_type = regrid_map[data]

        # extract dataarray from dataset, and wrap in a Variable (indicating current geo/time resolution is itself)
        var = Variable(var, name, name, time_regrid_type, geo_regrid_type)

        # save the variable to the pipeline namespace under the given identifier
        self.bind_value(name, var)


    #TODO: make this handle DataArrays instead of Datasets 
    def load_cmip6_data(self, variable:CMIP6Data, model:Model) -> xr.DataArray:
        """get an xarray with cmip6 data from the specified model"""

        all_realizations: list[xr.Dataset] = []
        for realization in self.realizations:
            all_scenarios: list[xr.Dataset] = []
            for scenario in self.scenarios:
        
                if model == Model.CAS_ESM2_0:
                    data = self.CAS_ESM2_0_cmip_loader(variable, realization, scenario)
                
                elif model == Model.FGOALS_f3_L:
                    data = self.FGOALS_f3_L_cmip_loader(variable, realization, scenario)

                #TODO: other models as needed
                else:
                    raise ValueError(f'Unrecognized model: {model}. Expected one of: {[*Model.__members__.values()]}')
            
                # add a scenario coordinate
                data = data.assign_coords(
                    ssp=('ssp', np.array([scenario.value], dtype='object'))
                )
                all_scenarios.append(data)
            
            # combine the scenario data into one xarray
            data = xr.concat(all_scenarios, dim='ssp')
            data = data.assign_coords(
                realization=('realization', np.array([realization.value], dtype='object'))
            )
            all_realizations.append(data)

        # combine all the data into one xarray with realization as a dimension
        dataset = xr.concat(all_realizations, dim='realization')

        return dataset[variable.value]


    #TODO: make this return a DataArray instead of a Dataset
    @staticmethod
    def CAS_ESM2_0_cmip_loader(variable: CMIP6Data, realization: Realization, scenario: Scenario) -> xr.Dataset:#xr.DataArray:
        """Data loader for the CAS-ESM2-0 model"""
        return xr.open_dataset(f'data/cmip6/{variable}/{variable}_Amon_CAS-ESM2-0_{scenario}_{realization}_gn_201501-210012.nc')#[variable.value]

    #TODO: make this return a DataArray instead of a Dataset
    @staticmethod
    def FGOALS_f3_L_cmip_loader(variable: CMIP6Data, realization: Realization, scenario: Scenario) -> xr.Dataset:#xr.DataArray:
        """Data loader for the FGOALS-f3-L model"""        
        return xr.open_dataset(f'data/cmip6/{variable}/{variable}_Amon_FGOALS-f3-L_{scenario}_{realization}_gr_201501-210012.nc')#[variable.value]

  
    @staticmethod
    def get_population_data(scenarios: list[Scenario]) -> xr.DataArray:
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
                        time=DatetimeNoLeap(year, 1, 1),
                        ssp=('ssp', np.array([scenario.value], dtype='object')) #note for population, only the first number is relevant
                    )
                all_years.append(data)

            # combine the scenario data into one xarray
            data = xr.concat(all_years, dim='time')
            all_scenarios.append(data)

        # combine all the data into one xarray with ssp as a dimension
        dataset = xr.concat(all_scenarios, dim='ssp')

        # # DEBUG crop to a smaller area
        # cropped_dataset = dataset.sel(lat=slice(30, -10, 1), lon=slice(-10, 30))
        # return cropped_dataset

        return dataset['population']


    @staticmethod
    def get_land_use_data() -> xr.Dataset:
        modis = xr.open_dataset('data/MODIS/land-use-5km.nc')
        modis = modis['LC_Type1']
        modis['time'] = modis.indexes['time'].to_datetimeindex().map(lambda dt: DatetimeNoLeap(dt.year, dt.month, dt.day))
        
        #TODO: handling modis over time? for now just take a single frame
        modis = modis.isel(time=0).drop(['time', 'crs'])

        return modis



    #TODO: other models' data loaders as needed


    @compile
    def threshold(self, y:ResultID, x:OperandID, /, threshold:Threshold):
        """
        Threshold a dataset, e.g. `y = x > threshold` or `y = x <= threshold`
        """

        #first make sure the data matches the specified resolution and frequency
        x = self.auto_regrid(x, allow_no_target=True)
        
        # perform the threshold operation
        var = self.get_value(x)
        
        if threshold.type == ThresholdType.greater_than:
            result = var.data > threshold.value
        elif threshold.type == ThresholdType.less_than:
            result = var.data < threshold.value
        elif threshold.type == ThresholdType.greater_than_or_equal:
            result = var.data >= threshold.value
        elif threshold.type == ThresholdType.less_than_or_equal:
            result = var.data <= threshold.value
        elif threshold.type == ThresholdType.equal:
            result = var.data == threshold.value
        elif threshold.type == ThresholdType.not_equal:
            result = var.data != threshold.value
        else:
            raise ValueError(f'Unrecognized threshold type: {threshold.type}. Expected one of: {[*ThresholdType.__members__.values()]}')

        # save the result to the pipeline namespace
        self.bind_value(y, Variable.from_result(result, var))



    @compile
    def fixed_time_regrid(self, y:ResultID, x:OperandID, /, target:Frequency):
        """
        regrid the given data to the given fixed temporal frequency
        
        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to regrid
            target (Frequency): the fixed target temporal frequency to regrid to, e.g. Frequency.monthly
        """
        raise NotImplementedError()
    
    @compile
    def matched_time_regrid(self, y:ResultID, x:OperandID, /, target:str):
        """
        regrid the given data to match the temporal frequency of the specified target data
        
        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to regrid
            target (str): the identifier for the target data to match temporal frequency with while regridding
                should target the resolution of an existing dataset in the pipeline by name, e.g. 'tasmax'
        """
        var = self.get_value(x)
        target_var = self.get_value(target)

        if var.time_regrid_type == TimeRegridType.none:
            #skip regridding (i.e. data has no time dimension)
            self.bind_value(y, var)
            return

        if var.time_regrid_type != TimeRegridType.interp:
            raise NotImplementedError(f'other time regrid types not yet implemented. Got: {var.time_regrid_type}')
        
        #TODO: handle warning when doing the interpolation
        new_data = var.data.interp({'time': target_var.data.time})
        var = Variable.from_result(new_data, var)
        var.frequency = target_var.frequency
        self.bind_value(y, var)


    @compile
    def fixed_geo_regrid(self, y:ResultID, x:OperandID, /, target:Resolution):
        """
        regrid the given data to the given fixed geo resolution

        Args:
            result (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to regrid
            target (Resolution): the fixed target geo resolution to regrid to, e.g. Resolution(0.5, 0.5),
        """
        raise NotImplementedError()
    
    @compile
    def matched_geo_regrid(self, y:ResultID, x:OperandID, /, target:str):
        """
        regrid the given data to match the geo resolution of the specified target data

        Args:
            result (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to regrid
            target (str): the identifier for the target data to match geo resolution while regridding
                should target the resolution of an existing dataset in the pipeline by name, e.g. 'tasmax'
        """
        var = self.get_value(x)
        target_var = self.get_value(target)


        # TODO: this is maybe a bit restrictive. perhaps we could just deal with when the coordinates are not in the same order
        assert var.data.dims[-2:] == ('lat', 'lon'), f'Variable "{x}" must have final dimensions (lat, lon) to be regridded'
        assert target_var.data.dims[-2:] == ('lat', 'lon'), f'Variable "{target}" must have final dimensions (lat, lon) to be used as a regrid target'

        # pull out all the values we need
        old_lats: np.ndarray = var.data.lat.data
        old_lons: np.ndarray = var.data.lon.data
        new_lats: np.ndarray = target_var.data.lat.data
        new_lons: np.ndarray = target_var.data.lon.data
        data: np.ndarray = var.data.data
        method = 'mean' #TODO: pull this from the geo_regrid_type

        # validate latitudes
        validate_latitudes(old_lats)
        validate_latitudes(new_lats)

        # ensure old_longitude convention ([-180,180] vs [0,360]) matches new_longitude. This also validates longitudes
        old_lon_convention = determine_longitude_convention(old_lons)
        new_lon_convention = determine_longitude_convention(new_lons)
        if old_lon_convention != new_lon_convention:
            old_lons = convert_longitude_convention(old_lons, new_lon_convention)

        # pdb.set_trace()
        # new_data = regrid_dataframe(data, {'lat_column': 'lat', 'lon_column':'lon'}, ['time'], 0.5,)

        #TODO: need to handle interpolation/other cases
        if var.geo_regrid_type == GeoRegridType.interp:
            old_data = xr.DataArray(
                data,
                dims=var.data.dims,
                coords={
                    **var.data.coords,
                    'lat': old_lats,
                    'lon': old_lons,
                }
            )
            new_data = old_data.interp({'lat': target_var.data.lat, 'lon': target_var.data.lon})
            var = Variable.from_result(new_data, var)
            var.resolution = target_var.resolution
            self.bind_value(y, var)
            return 

        if var.geo_regrid_type != GeoRegridType.sum:
            raise NotImplementedError(f'other geo regrid types not yet implemented. Got: {var.geo_regrid_type}')

        # create bin boundaries for the new latitudes and longitudes
        #TODO: break out bin creation into functions
        #TODO: bin creation needs to consider current lat/lon values. e.g. tasmax lon values are already binned, just need to add 360 as upper bound
        # new_lat_bins = create_lat_bins(new_lats)
        # new_lon_bins = create_lon_bins(new_lons)
        
        new_lat_delta = new_lats[1] - new_lats[0]
        new_lat_bins = np.concatenate((new_lats, [new_lats[-1] + new_lat_delta])) - new_lat_delta/2
        new_lon_delta = new_lons[1] - new_lons[0]
        new_lon_bins = np.concatenate((new_lons, [new_lons[-1] + new_lon_delta])) - new_lon_delta/2

        # add small epsilon to first and last elements to ensure that the bounds are inclusive
        new_lat_bins[0] += -np.sign(new_lat_delta) * 1e-10
        new_lat_bins[-1] += np.sign(new_lat_delta) * 1e-10
        new_lon_bins[0] += -np.sign(new_lon_delta) * 1e-10
        new_lon_bins[-1] += np.sign(new_lon_delta) * 1e-10

        # Find the corresponding bins for latitudes and longitudes
        lat_idx = np.digitize(old_lats, new_lat_bins) - 1
        lon_idx = np.digitize(old_lons, new_lon_bins) - 1

        #crop the data so that anything outside the new lat/lon bounds is discarded
        lat_idx_mask = (lat_idx >= 0) & (lat_idx < len(new_lats))
        lon_idx_mask = (lon_idx >= 0) & (lon_idx < len(new_lons))
        lat_idx = lat_idx[lat_idx_mask]
        lon_idx = lon_idx[lon_idx_mask]
        data = data[..., lat_idx_mask, :][..., lon_idx_mask]

        # Initialize the new data grid
        new_data = np.zeros(data.shape[:-2] + (len(new_lats), len(new_lons)), dtype=data.dtype)
        nan_accumulation = np.zeros(data.shape[:-2] + (len(new_lats), len(new_lons)), dtype=bool)

        #set any data that is nan to 0, and keep track of where the nans were
        validmask = ~np.isnan(data)
        data[~validmask] = 0

        # Accumulate over the binned indices, leaving non-geo dimensions alone
        idx = [np.arange(s) for s in data.shape[:-2]] + [lat_idx, lon_idx]
        mesh = np.meshgrid(*idx, indexing='ij')
        np.add.at(new_data, tuple(mesh), data)

        # Accumulate number of valid values per new cell    . anywhere that didn't have any valid values should be nan
        np.add.at(nan_accumulation, tuple(mesh), validmask)
        new_data[nan_accumulation == 0] = np.nan #TODO: perhaps if original data doesn't contain any nans, don't need this step?

        # convert new_data into an xarray
        new_data = xr.DataArray(
            new_data,
            dims=var.data.dims,
            coords={
                **var.data.coords,
                'lat': new_lats,
                'lon': new_lons,
            }
        )

        # save the result to the pipeline namespace
        var = Variable.from_result(new_data, var)
        var.resolution = target_var.resolution
        self.bind_value(y, var)
 

    def auto_regrid(self, x:OperandID, allow_no_target:bool=False) -> OperandID:
        """
        automatically regrid the given data to the target geo and temporal resolution of the pipeline

        Args:
            x (str): the identifier of the data to regrid
            allow_no_target (bool, optional): whether the pipeline may skip target geo/temporal resolutions. Defaults to False.

        Returns:
            str: the identifier of the regridded data
        """

        # check if geo/time targets are necessary
        if not allow_no_target:
            assert self.resolution is not None, f'Pipeline must have a target geo resolution before performing regrid operation on "{x}"'
            assert self.frequency is not None, f'Pipeline must have a target temporal resolution before performing regrid operation on "{x}"'

        #TODO: need to do regridding operations that reduce size of data before ones that increase it
        # perform the time regrid
        if self.frequency is not None and self.env[x].frequency != self.frequency:
            tmp_id = self._next_tmp_id()
            if isinstance(self.frequency, Frequency):
                self.fixed_time_regrid.unwrapped(tmp_id, x, self.frequency)
            else:
                self.matched_time_regrid.unwrapped(tmp_id, x, self.frequency)
            x = tmp_id
        
        # perform the geo regrid
        if self.resolution is not None and self.env[x].resolution != self.resolution:
            tmp_id = self._next_tmp_id()
            if isinstance(self.resolution, Resolution):
                self.fixed_geo_regrid.unwrapped(tmp_id, x, self.resolution)
            else:
                self.matched_geo_regrid.unwrapped(tmp_id, x, self.resolution)
            x = tmp_id

        return x
    

    @compile
    def multiply(self, y:ResultID, x1:OperandID, x2:OperandID, /):
        """
        Multiply two datasets together, i.e. `y = x1 * x2`

        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x1 (str): the identifier of the left operand
            x2 (str): the identifier of the right operand
        """

        #ensure data matches the specified resolution and frequency
        x1 = self.auto_regrid(x1)
        x2 = self.auto_regrid(x2)

        # perform the multiplication
        var1 = self.get_value(x1)
        var2 = self.get_value(x2)
        result = var1.data * var2.data

        # save the result to the pipeline namespace
        self.bind_value(y, Variable.from_result(result, var1))
    

    @compile
    def country_split(self, y:ResultID, x:OperandID, /, countries:list[str]):
        """
        Adds a new 'country' dimension to the data, and separates splits out the data at each given country

        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to split
            countries (list[str]): the list of countries to split the data by
        """

        var = self.get_value(x)

        data = var.data
        lat: np.ndarray = data.lat.values
        lon: np.ndarray = data.lon.values
        lon_grid, lat_grid = np.meshgrid(lon, lat)


        # ensure data has correct coordinate system
        # data = data.rio.write_crs(4326)



        # if no countries are specified, use all of them
        if countries is None:
            countries = self.sf['NAME_0'].unique().tolist()

        # get the shapefile rows for the countries we want
        countries_set = set(countries)
        countries_shp = self.sf[self.sf['NAME_0'].isin(countries_set)]

        # sort the countries in countries_shp to match the order of the countries in the data
        countries_shp = countries_shp.set_index('NAME_0').loc[countries].reset_index()

        # set up new np array of shape [len(countries), *data.shape]
        out_data = np.zeros((len(countries), *data.shape), dtype=data.dtype)
        
        for i, (_, country, gid, geometry) in enumerate(countries_shp.itertuples()):
            print(f'processing {country}...')

            # Generate a mask for the current country
            transform = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), len(lon), len(lat))
            mask = geometry_mask([geometry], transform=transform, invert=True, out_shape=(len(lat), len(lon)))

            # apply the mask to the data
            masked_data = data.where(xr.DataArray(mask, coords={'lat':lat, 'lon':lon}, dims=['lat', 'lon']))

            # insert the masked data into the output array
            out_data[i] = masked_data

        # combine all the data into a new DataArray with a country dimension
        out_data = xr.DataArray(
            out_data,
            dims=('country', *data.dims),
            coords={
                'country': countries,
                **data.coords,
            }
        )
        
        # save the result to the pipeline namespace
        self.bind_value(y, Variable.from_result(out_data, var))

    @compile
    def sum(self, y:ResultID, x:OperandID, /, dims:list[str]):
        """
        Sum the data along the given dimensions

        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to sum
            dims (list[str]): the list of dimensions to sum over
        """
        var = self.get_value(x)
        result = var.data.sum(dim=dims)
        self.bind_value(y, Variable.from_result(result, var))      


    @compile
    def save(self, x:OperandID, /, path:str):
        """
        Save the data to the given path

        Args:
            x (str): the identifier of the data to save
            path (str): the path to save the data to
        """
        var = self.get_value(x)
        var.data.to_netcdf(path)


    def execute(self):
        """Execute the pipeline"""
        for func, args, kwargs in self.steps:
            func(*args, **kwargs)

    @property
    def sf(self):
        """Get the country shapefile"""
        if self._sf is None:
            print(f'Loading country shapefile...')
            self._sf = gpd.read_file('gadm_0/gadm36_0.shp')
        return self._sf


def get_available_cmip6_data() -> list[tuple[CMIP6Data, Model, Scenario, Realization]]:
    # parse all cmip6 data filepaths, and make a list of what is available (variable, model, scenario, realization)
    pdb.set_trace()
    ...




def heat_scenario():


    pipe = Pipeline(
        realizations=Realization.r1i1p1f1, 
        scenarios=[Scenario.ssp126, Scenario.ssp245, Scenario.ssp370, Scenario.ssp585],
    )
    
    # e.g. target fixed geo/temporal resolutions
    # pipe.set_geo_resolution(Resolution(0.5, 0.5))
    # pipe.set_time_resolution(Frequency.monthly)

    # e.g. target geo/temporal resolution of existing data in pipeline
    # pipe.set_geo_resolution('pop')
    pipe.set_geo_resolution('tasmax')
    # pipe.set_time_resolution('tasmax')
    pipe.set_time_resolution('pop')

    # load the data
    pipe.load('pop', OtherData.population)
    pipe.load('tasmax', CMIP6Data.tasmax, Model.CAS_ESM2_0)
    
    # operations on the data to perform the scenario
    pipe.threshold('heat', 'tasmax', Threshold(308.15, ThresholdType.greater_than))
    pipe.multiply('exposure0', 'heat', 'pop')
    pipe.country_split('exposure1', 'exposure0', ['China', 'India', 'United States', 'Canada', 'Mexico', 'Brazil', 'Australia'])
    pipe.sum('exposure2', 'exposure1', dims=['lat', 'lon'])
    pipe.save('exposure2', 'exposure.nc')

    # run the pipeline
    pipe.execute()

    # e.g. extract any live results
    res = pipe.get_last_value()
    # pop = pipe.get_value('pop')
    # tasmax = pipe.get_value('tasmax')

    # plot all the countries on a single plot
    for country in res.data['country'].values:
        res.data.sel(country=country).isel(ssp=0).plot(label=country)

    plt.title('People Exposed to Heatwaves by Country')
    plt.legend()
    plt.show()




def crop_scenario():
    pipe = Pipeline(
        realizations=Realization.r1i1p1f1,
        scenarios=[Scenario.ssp126, Scenario.ssp245, Scenario.ssp370, Scenario.ssp585],
    )
    pipe.set_geo_resolution(Resolution(0.5, 0.5))
    pipe.set_time_resolution(Frequency.monthly)
    pipe.load('tas', CMIP6Data.tas, Model.FGOALS_f3_L)
    pipe.load('pr', CMIP6Data.pr, Model.FGOALS_f3_L)
    pipe.load('land_cover', OtherData.land_cover)

    #TODO: rest of scenario


def demo_scenario():
    pipe = Pipeline(
        realizations=Realization.r1i1p1f1,
        scenarios=Scenario.ssp585
    )
    pipe.set_geo_resolution('modis')
    pipe.set_time_resolution('pop')
    pipe.load('modis', OtherData.land_cover)
    pipe.load('pop', OtherData.population)
    pipe.load('tasmax', CMIP6Data.tasmax, Model.CAS_ESM2_0)
    pipe.threshold('heat', 'tasmax', Threshold(308.15, ThresholdType.greater_than))
    pipe.multiply('exposure0', 'heat', 'pop')
    pipe.threshold('urban_mask', 'modis', Threshold(13, ThresholdType.equal))
    pipe.threshold('not_urban_mask', 'modis', Threshold(13, ThresholdType.not_equal))
    pipe.multiply('urban_exposure', 'exposure0', 'urban_mask')
    pipe.multiply('not_urban_exposure', 'exposure0', 'not_urban_mask')
    pipe.sum('global_urban_exposure', 'urban_exposure', dims=['lat', 'lon'])
    pipe.sum('global_not_urban_exposure', 'not_urban_exposure', dims=['lat', 'lon'])
    pipe.save('global_urban_exposure', 'global_urban_exposure.nc')
    pipe.save('global_not_urban_exposure', 'global_not_urban_exposure.nc')

    pipe.execute()

    pdb.set_trace()
    ...


if __name__ == '__main__':
    # heat_scenario()
    # crop_scenario()
    demo_scenario()