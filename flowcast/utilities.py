from __future__ import annotations

from typing import Callable, TypeVar, Any, Iterable
from typing_extensions import ParamSpec

import inspect
import ast
import textwrap

import pdb

_P = ParamSpec('_P')
_R_co = TypeVar('_R_co', covariant=True)

def method_uses_prop(cls:type, method:Callable[_P, _R_co], prop:str, key:Callable[[Any], Callable[_P, _R_co]]=lambda m:m) -> bool:
    """
    Check if a method access a given property of the class in its source code.
    So far this is mainly used to check if a pipeline method requires GADM data which might not be downloaded yet.

    Args:
    - cls (type): The class that the method is attached to
    - method (callable): The method to check
    - prop (str): The name of the property to check for
    - key (callable): used for accessing the method directly from the `cls.__dict__`. usage is `key(cls.__dict__[method.__name__])`. E.g. pipeline methods are wrapped, so key can be used to unwrap them. Defaults to the identity function.
    """

    #check if the method is static. Static methods don't use properties
    class_attached = key(cls.__dict__[method.__name__])
    if isinstance(class_attached, staticmethod):
        return False

    # Fetch the source code of the method
    source = inspect.getsource(method)

    # get the parameter that represents self
    sig = inspect.signature(method)
    params = list(sig.parameters.values())
    assert len(params) > 0, f'Expected at least one parameter in method signature. Got: {sig}'
    self_param = params[0]
    
    # Dedent the source code to remove leading whitespaces
    dedented_source = textwrap.dedent(source)
    
    # Parse the dedented source code to get its AST
    tree = ast.parse(dedented_source)
    
    # Traverse the AST to look for instances where property 'a' of 'self' is accessed
    for node in ast.walk(tree):
        if isinstance(node, ast.Attribute) and node.attr == prop and isinstance(node.value, ast.Name) and node.value.id == self_param.name:
            return True
    return False


T = TypeVar('T')
def partition(elements:Iterable[T], condition:Callable[[T], bool]) -> tuple[list[T], list[T]]:
    """
    Partition a list of elements into two lists based on a condition.

    Args:
    - elements (iterable): The list of elements to partition
    - condition (callable): The condition to partition the elements on

    Returns:
    - tuple: A tuple containing two lists. The first list contains elements that satisfy the condition, the second list contains elements that don't satisfy the condition.
    """
    true_list = []
    false_list = []
    for element in elements:
        if condition(element):
            true_list.append(element)
        else:
            false_list.append(element)
    return (true_list, false_list)

import xarray as xr
import numpy as np
from scipy import stats
def nanmode(data:np.ndarray, dims:int|list[int]|None=None) -> np.ndarray:
    """
    Take the mode of the data along the specified dimensions

    Args:
    - data (np.ndarray): The data to take the mode of
    - dims (int|list[int], optional): The dimension(s) to take the mode over. If None, all dimensions are used. Defaults to None.

    Returns:
    - np.ndarray: The mode of the data along the specified dimensions
    """

    # ensure dims is a list
    if dims is None:
        dims = list(range(len(data.shape)))
    elif isinstance(dims, int):
        dims = [dims]

    # convert any negative dimensions to positive
    dims = [dim if dim >= 0 else len(data.shape) + dim for dim in dims]

    #transpose so that the dimensions to take the mode over are the last dimensions
    # reduce_dims, keep_dims = partition(var.data.dims, lambda dim: dim in dims)
    keep_dims = [i for i in range(len(data.shape)) if i not in dims]
    reduce_dims = dims
    data = np.transpose(data, keep_dims+reduce_dims)
    data = data.reshape(*(data.shape[:-len(reduce_dims)]+ (-1,)))

    # keep track of where all values in the mode slice are NaN
    # This is necessary because scipy 1.10 stats.mode would set a slice of all NaNs to 0
    # TODO: want to upgrade to scipy >= 1.11 since it properly handles all NaN slices 
    nan_mask = np.all(np.isnan(data), axis=-1)

    # take the mode
    data = np.ascontiguousarray(data)
    data = stats.mode(data, axis=-1, nan_policy='omit', keepdims=False).mode

    # data might not be an array
    # e.g. if the mode is a scalar, 
    # or mode had NaNs which returns some weird Masked Array type
    # TODO: TBD if this is the same in scipy >= 1.11
    data = np.array(data)

    # set all NaN slices back to NaN
    data[nan_mask] = np.nan

    return data

def xarray_mode(data:xr.DataArray, dims:str|list[str]|None=None) -> xr.DataArray:
    """
    Take the mode of the xarray DataArray along the specified dimensions

    Args:
    - data (xr.DataArray): The data to take the mode of
    - dims (list[str], optional): The dimensions to take the mode over. If None, all dimensions are used. Defaults to None.
    """
    # ensure dims is a list
    if dims is None:
        dims = [*data.dims]
    elif isinstance(dims, str):
        dims = [dims]

    # get the indices of the dimensions to reduce and the dimensions to keep
    reduce_dims, keep_dims = partition(data.dims, lambda dim: dim in dims)
    reduce_dims = [data.dims.index(dim) for dim in reduce_dims]
    keep_dims = [data.dims.index(dim) for dim in keep_dims]

    # take the mode
    result = nanmode(data.data, dims=reduce_dims)

    # convert result to xarray DataArray
    result = xr.DataArray(result, coords=[data.coords[i] for i in keep_dims], dims=[data.dims[i] for i in keep_dims])

    return result