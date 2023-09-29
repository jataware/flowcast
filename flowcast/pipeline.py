from __future__ import annotations

from matplotlib import pyplot as plt
import numpy as np
import xarray as xr
import geopandas as gpd
from rasterio.transform import from_bounds
from rasterio.features import geometry_mask


from .spacetime import Frequency, Resolution, DatetimeNoLeap, LongitudeConvention, inplace_set_longitude_convention
from .regrid import RegridType, regrid_1d
from .utilities import method_uses_prop, setup_gadm

from os.path import dirname, abspath
from itertools import count
from inspect import signature, Signature
from enum import Enum, auto
from typing import Any, Callable, TypeVar, Mapping #, Protocol
from typing_extensions import ParamSpec
from types import MethodType
from dataclasses import dataclass


# pretty printing
from rich import print, traceback; traceback.install(show_locals=True)


import pdb




class ThresholdType(Enum):
    greater_than = auto()
    less_than = auto()
    greater_than_or_equal = auto()
    less_than_or_equal = auto()
    equal = auto()
    not_equal = auto()

@dataclass
class Threshold:
    value: float|int
    type: ThresholdType


# Datatypes for representing identifiers in the pipeline namespace
class ResultID(str): ...
class OperandID(str): ...


@dataclass
class Variable:
    data: xr.DataArray
    time_regrid_type: RegridType|None
    geo_regrid_type: RegridType|None

@dataclass
class PipelineVariable(Variable):
    frequency: Frequency|str
    resolution: Resolution|str
    
    @staticmethod
    def from_result(
        data:xr.DataArray,
        prev:'PipelineVariable',
        *,
        time_regrid_type:RegridType=None,
        geo_regrid_type:RegridType=None,
        frequency:Frequency|str=None,
        resolution:Resolution|str=None
    ) -> 'PipelineVariable':
        """Create a new Variable from the given data, inheriting the geo/temporal resolution from a previous Variable"""
        return PipelineVariable(
            data,
            time_regrid_type or prev.time_regrid_type,
            geo_regrid_type or prev.geo_regrid_type,
            frequency or prev.frequency,
            resolution or prev.resolution
        )
    
    @classmethod
    def from_base_variable(cls, var:Variable, frequency:Frequency|str, resolution:Resolution|str) -> 'PipelineVariable':
        return cls(var.data, var.time_regrid_type, var.geo_regrid_type, frequency, resolution)
    

_R_co = TypeVar("_R_co", covariant=True)
_P = ParamSpec("_P")

# class CallableWithUnwrapped(Protocol[_P, _R_co], Callable[_P, _R_co]):
#     # def __call__(self, *args: _P.args, **kwargs: _P.kwargs) -> _R_co: ...
    
#     @property
#     def unwrapped(self) -> Callable[_P, _R_co]: ...

class Pipeline:


    def __init__(self, *, verbose:bool=True, low_memory:bool=False):
        
        # dynamic settings for data to be used in pipeline
        self.resolution: Resolution|str|None = None
        self.frequency: Frequency|str|None = None
        
        # list of steps in the pipeline DAG
        self.steps: list[tuple[MethodType, tuple[Any, ...]]] = []

        # keep track of the most recent result during pipeline execution
        self.last_set_identifier: str|None = None

        # namespace for binding operation results to identifiers 
        self.env: dict[str, PipelineVariable] = {}

        # keep track of all identifiers while the pipeline is being compiled
        self.compiled_ids: set[str] = set()

        # tmp id counter
        self.tmp_id_counter = count(0)

        # whether to print out debug info during pipeline execution
        self.verbose = verbose

        # whether to use low memory mode
        self.low_memory = low_memory

        # bind the current instance to all unwrapped compiled methods (so we don't need to pass the instance manually)
        for attr_name, attr_value in vars(Pipeline).items():
            if hasattr(attr_value, "unwrapped"):
                bound_unwrapped = attr_value.unwrapped.__get__(self, Pipeline)
                setattr(attr_value, "unwrapped", bound_unwrapped)

        # shapefile for country data. Only load if needed
        self._sf = None


    def compile(method:Callable[_P, _R_co]) -> Callable[_P, None]:
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

            # check if the method uses sf, meaning we need to verify gadm shapefiles are available
            if method_uses_prop(Pipeline, method, 'sf', key=lambda m: m.unwrapped):
                setup_gadm()

        # save the unmodified original function in for use inside pipeline methods
        wrapper.unwrapped = method

        return wrapper

    
    @staticmethod
    def step_str(step:tuple[MethodType, tuple[Any, ...], dict[str, Any]]):
        """get the repr for the given step in the pipeline"""
        func, args, kwargs = step
        args = args[1:] #skip self. Removing this causes infinite recursion
        name = func.__name__
        args_str = ', '.join([str(arg) for arg in args])
        kwargs_str = ', '.join([f'{key}={value}' for key, value in kwargs.items()])
        return f'Pipeline.{name}({", ".join((args_str, kwargs_str))})'
    

    def __str__(self):
        """get the str for the pipeline"""
        return '\n'.join([f'{i}: {Pipeline.step_str(step)}' for i, step in enumerate(self.steps)])

    def __repr__(self):
        """get the repr for the pipeline"""
        return f'Pipeline({[func.__name__ for func, _, _ in self.steps]})'

    
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
            raise ValueError(f'Operand identifier "{identifier}" does not exist in pipeline at step {len(self.steps)-1}: {self.step_str(self.steps[-1])}')
    
    def _assert_id_is_unique_and_mark_used(self, identifier:str):
        """
        (Compile-time) assert that the given identifier is unique, and add it to the compiled_ids set
        Should be called inside any compile-time functions that will add a variable to the pipeline namespace
        NOTE: always call after appending the current step to the pipeline

        @compile handles this automatically
        """
        if identifier in self.compiled_ids:
            raise ValueError(f'Tried to reuse "{identifier}" for result identifier on step {len(self.steps)-1}: {self.step_str(self.steps[-1])}. All identifiers must be unique.')
        self.compiled_ids.add(identifier)

    
    
    def bind_value(self, identifier:ResultID, value:PipelineVariable):
        """Bind a value to an identifier in the pipeline namespace"""
        assert identifier not in self.env, f'Identifier "{identifier}" already exists in pipeline namespace. All identifiers must be unique.'
        self.last_set_identifier = identifier
        self.env[identifier] = value


    def get_value(self, identifier:OperandID) -> PipelineVariable:
        """Get a value from the pipeline namespace"""
        return self.env[identifier]
    

    def get_last_value(self) -> PipelineVariable:
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
    def load(self, name:ResultID, /, loader:Callable[[], Variable]):
        """
        Load existing data into the pipeline

        Args:
            identifier (str): the identifier to bind the data to in the pipeline namespace
            data (enum): the data to load
            model (enum, optional): the model to load the data from (required for CMIP6 data)
        """
        var = loader()

        #ensure the lon conventions match
        if 'lon' in var.data.coords:
            inplace_set_longitude_convention(var.data, LongitudeConvention.neg180_180)
        #TODO: match the latitude convention

        self.bind_value(name, PipelineVariable.from_base_variable(var, name, name))


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
        self.bind_value(y, PipelineVariable.from_result(result, var))



    @compile
    def fixed_time_regrid(self, y:ResultID, x:OperandID, /, target:Frequency):
        """
        regrid the given data to the given fixed temporal frequency
        
        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to regrid
            target (Frequency): the fixed target temporal frequency to regrid to, e.g. Frequency.monthly
        """
        var = self.get_value(x)
        if var.frequency == target or var.time_regrid_type is None:
            #skip regridding
            self.bind_value(y, var)
            return
        
        # generate the new time coordinates
        min_time = var.data.time.data.min()
        max_time = var.data.time.data.max()
        if target == Frequency.monthly:
            min_time = min_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            max_time = max_time.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            times = np.array([DatetimeNoLeap(year, month, 1) for year in range(min_time.year, max_time.year+1) for month in range(1, 13)])
        elif target == Frequency.yearly:
            min_time = min_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            max_time = max_time.replace(month=1, day=1, hour=0, minute=0, second=0, microsecond=0)
            times = np.array([DatetimeNoLeap(year, 1, 1) for year in range(min_time.year, max_time.year+1)])
        elif target == Frequency.decadal:
            min_time = DatetimeNoLeap(min_time.year - min_time.year % 10, 1, 1)
            max_time = DatetimeNoLeap(max_time.year + max_time.year % 10, 1, 1)
            times = np.array([DatetimeNoLeap(year, 1, 1) for year in range(min_time.year, max_time.year+1, 10)])

        # regrid the data and save the result to the pipeline namespace
        new_data = regrid_1d(var.data, times, 'time', aggregation=var.time_regrid_type, low_memory=self.low_memory)
        var = PipelineVariable.from_result(new_data, var, frequency=target)
        self.bind_value(y, var)

        
        
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

        if var.frequency == target_var.frequency or var.time_regrid_type is None:
            #skip regridding for already matching data, and data has no time dimension
            self.bind_value(y, var)
            return

        new_data = regrid_1d(var.data, target_var.data.time.data, 'time', aggregation=var.time_regrid_type, low_memory=self.low_memory)
        var = PipelineVariable.from_result(new_data, var, frequency=target_var.frequency)
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
        var = self.get_value(x)
        if var.resolution == target or var.geo_regrid_type is None:
            #skip regridding
            self.bind_value(y, var)
            return

        # raw geo coordinates at target resolution
        lats = np.arange(-90, 90, target.dy)
        lons = np.arange(-180, 180, target.dx)

        # crop geo coordinates around the data's maximum extents
        min_lat = var.data.lat.data.min()
        max_lat = var.data.lat.data.max()
        min_lon = var.data.lon.data.min()
        max_lon = var.data.lon.data.max()
        lats = lats[(lats + target.dy/2 >= min_lat) & (lats - target.dy/2 <= max_lat)]
        lons = lons[(lons + target.dx/2 >= min_lon) & (lons - target.dx/2 <= max_lon)]

        # regrid the data and save the result to the pipeline namespace
        new_data = regrid_1d(var.data, lats, 'lat', aggregation=var.geo_regrid_type, low_memory=self.low_memory)
        new_data = regrid_1d(new_data, lons, 'lon', aggregation=var.geo_regrid_type, low_memory=self.low_memory)
        var = PipelineVariable.from_result(new_data, var, resolution=target)
        self.bind_value(y, var)
    
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

        new_data = regrid_1d(var.data, target_var.data.lat.data, 'lat', aggregation=var.geo_regrid_type, low_memory=self.low_memory)
        new_data = regrid_1d(new_data, target_var.data.lon.data, 'lon', aggregation=var.geo_regrid_type, low_memory=self.low_memory)
        var = PipelineVariable.from_result(new_data, var, resolution=target_var.resolution)
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
    

    def binop(self, y:ResultID, x1:OperandID, x2:OperandID, op:Callable[[xr.DataArray, xr.DataArray], xr.DataArray], /):
        """
        Perform a binary operation on two datasets, e.g. `y = x1 + x2`

        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x1 (str): the identifier of the left operand
            x2 (str): the identifier of the right operand
            op (Callable[[xr.DataArray, xr.DataArray], xr.DataArray]): the binary operation to perform
        """

        #ensure data matches the specified resolution and frequency
        x1 = self.auto_regrid(x1)
        x2 = self.auto_regrid(x2)

        # perform the operation
        var1 = self.get_value(x1)
        var2 = self.get_value(x2)
        result = op(var1.data, var2.data)

        # save the result to the pipeline namespace
        self.bind_value(y, PipelineVariable.from_result(result, var1))
    
    @compile
    def add(self, y:ResultID, x1:OperandID, x2:OperandID, /):
        """
        Add two datasets together, i.e. `y = x1 + x2`

        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x1 (str): the identifier of the left operand
            x2 (str): the identifier of the right operand
        """
        self.binop(y, x1, x2, lambda x1, x2: x1 + x2)


    @compile
    def subtract(self, y:ResultID, x1:OperandID, x2:OperandID, /):
        """
        Subtract two datasets, i.e. `y = x1 - x2`

        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x1 (str): the identifier of the left operand
            x2 (str): the identifier of the right operand
        """
        self.binop(y, x1, x2, lambda x1, x2: x1 - x2)


    @compile
    def multiply(self, y:ResultID, x1:OperandID, x2:OperandID, /):
        """
        Multiply two datasets together, i.e. `y = x1 * x2`

        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x1 (str): the identifier of the left operand
            x2 (str): the identifier of the right operand
        """
        self.binop(y, x1, x2, lambda x1, x2: x1 * x2)


    @compile
    def divide(self, y:ResultID, x1:OperandID, x2:OperandID, /):
        """
        Divide two datasets, i.e. `y = x1 / x2`

        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x1 (str): the identifier of the left operand
            x2 (str): the identifier of the right operand
        """
        self.binop(y, x1, x2, lambda x1, x2: x1 / x2)

    @compile
    def isel(self, y:ResultID, x:OperandID, /, indexers:Mapping[Any, Any]|None=None, drop:bool=False):
        """
        Perform an isel operation on the given data. 
        See: https://xarray.pydata.org/en/stable/generated/xarray.DataArray.isel.html

        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to perform the isel operation on
            indexers (Mapping[Any, Any], optional): A dict with keys matching dimensions and values given by integers, slice objects or arrays. indexer can be a integer, slice, array-like or DataArray. Defaults to None.
            drop (bool, optional): Drop coordinates variables indexed by integers instead of making them scalar. Defaults to False.
        """
        var = self.get_value(x)
        result = var.data.isel(indexers=indexers, drop=drop)
        self.bind_value(y, PipelineVariable.from_result(result, var))

    # def where(self, y:ResultID, x:OperandID, cond:OperandID, /):
    # def cat(self, y:ResultID, x:OperandID, /, dim:str): coord np.ndarray data could be optional
    # def stack(self, y:ResultID, x:OperandID, /, coord:{name:str, data:np.ndarray}):

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
        
        # needed for geometry_mask. cache here since we don't need to recompute it for each country
        transform = from_bounds(lon.min(), lat.min(), lon.max(), lat.max(), len(lon), len(lat))

        for i, (_, country, gid, geometry) in enumerate(countries_shp.itertuples()):
            self.print(f'processing {country}...')

            # Generate a mask for the current country and apply the mask to the data
            mask = geometry_mask([geometry], transform=transform, invert=True, out_shape=(len(lat), len(lon)))
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
        self.bind_value(y, PipelineVariable.from_result(out_data, var))

    @compile
    def sum_reduce(self, y:ResultID, x:OperandID, /, dims:list[str]):
        """
        Sum the data along the given dimensions

        Args:
            y (str): the identifier to bind the result to in the pipeline namespace
            x (str): the identifier of the data to sum
            dims (list[str]): the list of dimensions to sum over
        """
        var = self.get_value(x)
        result = var.data.sum(dim=dims)
        self.bind_value(y, PipelineVariable.from_result(result, var))

    #def mean_reduce(self, y:ResultID, x:OperandID, /, dims:list[str]):
    #def max_reduce(self, y:ResultID, x:OperandID, /, dims:list[str]):
    #def min_reduce(self, y:ResultID, x:OperandID, /, dims:list[str]):
    #def std_reduce(self, y:ResultID, x:OperandID, /, dims:list[str]):
    #def median_reduce(self, y:ResultID, x:OperandID, /, dims:list[str]):
    #def mode_reduce(self, y:ResultID, x:OperandID, /, dims:list[str]):


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
            self.print(self.step_str((func, args, kwargs)))
            func(*args, **kwargs)

    @property
    def sf(self):
        """Get the country shapefile"""
        if self._sf is None:
            self.print(f'Loading country shapefile...')
            self._sf = gpd.read_file(f'{dirname(abspath(__file__))}/gadm_0/gadm36_0.shp')
        return self._sf
    
    def print(self, *args, **kwargs):
        if self.verbose:
            print(*args, **kwargs)
