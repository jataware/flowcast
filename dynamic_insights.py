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



def get_signature_from_source(method):
    source = getsource(method)
    match = re.search(r'def\s+\w+\((.*?)\):', source, re.DOTALL)
    if match:
        return match.group(1).strip()
    else:
        raise ValueError("Couldn't extract function signature from source")





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
    land_use = 'land_use'


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
    data: xr.DataArray
    frequency: Frequency|str
    resolution: Resolution|str

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

class ResultID(str): ...
class OperandID(str): ...

#TODO:
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
        match data:
            case OtherData.population:
                var = self.get_population_data(self.scenarios)
            case OtherData.land_use:
                var = self.get_land_use_data()
            case CMIP6Data():
                assert model is not None, 'Must specify a model for CMIP6 data'
                var = self.load_cmip6_data(data, model)
            case _:
                raise ValueError(f'Unrecognized data type: {data}. Expected one of: {CMIP6Data}, {OtherData}')

        # extract dataarray from dataset, and wrap in a Variable. 
        # set geo/time resolution to itself
        var = Variable(var[data.value], name, name)

        # save the variable to the pipeline namespace under the given identifier
        self.bind_value(name, var)


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
            data = data.assign_coords(
                realization=('realization', np.array([realization.value], dtype='object'))
            )
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


    @staticmethod
    def get_land_use_data() -> xr.Dataset:
        raise NotImplementedError()


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
        match threshold.type:
            case ThresholdType.greater_than:
                result = var.data > threshold.value
            case ThresholdType.less_than:
                result = var.data < threshold.value
            case ThresholdType.greater_than_or_equal_to:
                result = var.data >= threshold.value
            case ThresholdType.less_than_or_equal_to:
                result = var.data <= threshold.value
            case ThresholdType.equal_to:
                result = var.data == threshold.value
            case ThresholdType.not_equal_to:
                result = var.data != threshold.value

        # save the result to the pipeline namespace
        self.bind_value(y, Variable(result, var.frequency, var.resolution))



    @compile
    def time_regrid(self, y:ResultID, x:OperandID, /, target:Frequency|str):
        """
        regrid the given data to the given temporal frequency
        
        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to regrid
            target (Frequency|str): the target temporal frequency to regrid to.
                Can either be a fixed Frequency object, e.g. Frequency.monthly,
                or target the frequency of an existing dataset in the pipeline by name, e.g. 'tasmax'
        """
        raise NotImplementedError()
    

    @compile
    def geo_regrid(self, y:ResultID, x:OperandID, /, target:Resolution|str):
        """
        regrid the given data to the given geo resolution

        Args:
            result (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to regrid
            target (Resolution|str): the target geo resolution to regrid to.
                Can either be a fixed Resolution object, e.g. Resolution(0.5, 0.5),
                or target the resolution of an existing dataset in the pipeline by name, e.g. 'tasmax'
        """
        raise NotImplementedError()

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

        # perform the geo regrid
        if self.resolution is not None and self.env[x].resolution != self.resolution:
            tmp_id = self._next_tmp_id()
            self.geo_regrid.unwrapped(tmp_id, x, self.resolution)
            x = tmp_id

        # perform the time regrid
        if self.frequency is not None and self.env[x].frequency != self.frequency:
            tmp_id = self._next_tmp_id()
            self.time_regrid.unwrapped(tmp_id, x, self.frequency)
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
        self.bind_value(y, Variable(result, var1.frequency, var1.resolution))
    

    @compile
    def country_split(self, y:ResultID, x:OperandID, /, countries:list[str]):
        """
        Adds a new 'country' dimension to the data, and separates splits out the data at each given country

        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to split
            countries (list[str]): the list of countries to split the data by
        """
        raise NotImplementedError()
    

    @compile
    def sum(self, y:ResultID, x:OperandID, /, dims:list[str]):
        """
        Sum the data along the given dimensions

        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to sum
            dims (list[str]): the list of dimensions to sum over
        """
        raise NotImplementedError()


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

    # run the pipeline
    pipe.execute()

    # e.g. extract any live results
    # res = pipe.get_last_value()
    # pop = pipe.get_value('pop')
    # tasmax = pipe.get_value('tasmax')




def crop_scenario():
    pipe = Pipeline(
        realizations=Realization.r1i1p1f1,
        scenarios=[Scenario.ssp126, Scenario.ssp245, Scenario.ssp370, Scenario.ssp585],
    )
    pipe.set_geo_resolution(Resolution(0.5, 0.5))
    pipe.set_time_resolution(Frequency.monthly)
    pipe.load('tas', CMIP6Data.tas, Model.FGOALS_f3_L)
    pipe.load('pr', CMIP6Data.pr, Model.FGOALS_f3_L)
    pipe.load('land_use', OtherData.land_use)

    #TODO: rest of scenario


if __name__ == '__main__':
    heat_scenario()
    # crop_scenario()