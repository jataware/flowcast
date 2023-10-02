from __future__ import annotations

from typing import Callable, TypeVar, Any
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

    Parameters:
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