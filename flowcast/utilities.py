from __future__ import annotations

from typing import Callable, TypeVar
from typing_extensions import ParamSpec

import inspect
import ast
import random
import textwrap

import pdb

_P = ParamSpec('_P')
_R_co = TypeVar('_R_co', covariant=True)


def is_static(method:Callable[_P, _R_co]) -> bool:
    # If it's bound to an instance or class
    if hasattr(method, "__self__"):
        cls = method.__self__ if isinstance(method.__self__, type) else method.__self__.__class__
        return isinstance(cls.__dict__.get(method.__name__, None), staticmethod)

    # If it's unbound
    for global_cls in globals().values():
        if isinstance(global_cls, type) and hasattr(global_cls, method.__name__):
            global_method = global_cls.__dict__[method.__name__]
            if isinstance(global_method, staticmethod) and global_method.__func__ is method:
                return True
            elif global_method is method:
                return False
    
    pdb.set_trace()
    raise ValueError(f'Failed to identify the class of method {method}')


def method_uses_prop(method:Callable[_P, _R_co], prop:str) -> bool:

    # verify that the method is a method
    #TODO:...

    # static methods don't have access to self
    if is_static(method):
        return False

    # unbind the method if it is bound
    if hasattr(method, '__func__'):
        method = method.__func__


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



def setup_gadm():
    print('Setting up GADM')
    print('TODO: implement setup_gadm')



if __name__ == '__main__':
    # example usage
    class A:
        def __init__(self):
            self.a = 5
        
        def some_method(self):
            if random.random() > 0.5:
                print(f'{self.a=}')
            else:
                print('a not used in this branch')

        def another_method(belf):
            print('a not used in this method')
            belf.a

        @staticmethod
        def static_method(self):
            print('a not used in this method')
            A().a
            self.a


    print(method_uses_prop(A().some_method, 'a'))  # True
    print(method_uses_prop(A.another_method, 'a'))  # False
    print(method_uses_prop(A.static_method, 'a'))  # False
    print(method_uses_prop(A().static_method, 'a'))  # False
