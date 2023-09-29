from __future__ import annotations

from typing import Callable, TypeVar, Any
from typing_extensions import ParamSpec

import inspect
import ast
import textwrap

import pdb

_P = ParamSpec('_P')
_R_co = TypeVar('_R_co', covariant=True)

#TODO: move to pipeline.py
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


#TODO: rename file to gadm.py
import os
from os.path import dirname, abspath, isdir, join
import requests
from pathlib import Path
from zipfile import ZipFile

# default_gadm_path = Path(dirname(abspath(__file__)), 'gadm')
default_gadm_path = Path(Path(__file__).parent, 'gadm')

gadm_env_var = 'GADM_DIR'

from tqdm import tqdm

def verify_gadm(dir:Path):
    """Presuming that GADM data is already downloaded, verify that the given directory contains the GADM data in the expected format"""
    #verify that the directory exists
    assert isdir(dir), f'Invalid GADM directory. Expected a directory. Got: {dir}'
    #TODO: rest of verification
    pdb.set_trace()

def download_gadm(dir:Path):
    """Download GADM data to the given directory"""

    # create the directory if it doesn't exist
    os.makedirs(dir, exist_ok=True)

    # download admin2 and admin3 from the S3 bucket
    urls = [
        'https://jataware-world-modelers.s3.amazonaws.com/gadm/gadm36_2.feather.zip',
        'https://jataware-world-modelers.s3.amazonaws.com/gadm/gadm36_3.feather.zip'
    ]
    filenames = [url.split('/')[-1] for url in urls]

    for url, filename in zip(urls, filenames):
        # download the file
        r = requests.get(url, allow_redirects=True, stream=True)
        assert r.status_code == 200, f'Failed to download GADM data from {url}. Got status code: {r.status_code}'

        total_size = int(r.headers.get('content-length', 0))

        # save the file
        with open(join(dir, filename), 'wb') as f, tqdm(desc=f'download {filename}', total=total_size, unit='iB', unit_scale=True) as bar:
            for chunk in r.iter_content(chunk_size=1024):
                f.write(chunk)
                bar.update(len(chunk))

        # unzip the file with a progress bar
        with ZipFile(join(dir, filename), 'r') as z:
            # Get the total uncompressed size for the progress bar
            total_size = sum(item.file_size for item in z.infolist())
            with tqdm(desc=f'unzip {filename}', total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as bar:
                
                for member in z.infolist():
                    # Extract each member in chunks to update the progress bar
                    with z.open(member, 'r') as source, open(Path(dir, member.filename), 'wb') as target:
                        chunk_size = 1024 * 1024  # 1MB
                        while True:
                            chunk = source.read(chunk_size)
                            if not chunk:
                                break
                            target.write(chunk)
                            bar.update(len(chunk))


        # delete the zip file
        os.remove(join(dir, filename))



def setup_gadm():
    """Ensure that GADM data is downloaded and ready to use"""
    # if GADM_DIR environment variable is set, use that
    gadm_dir = os.environ.get(gadm_env_var)
    if gadm_dir is not None:
        gadm_dir = Path(gadm_dir)
        verify_gadm(gadm_dir)
        print(f'Using ENV specified GADM data from {gadm_dir}')
        return

    # otherwise use the default directory, and download GADM if necessary
    try:
        verify_gadm(default_gadm_path)
        print(f'Using default GADM data from {default_gadm_path}')
        return
    except:
        print(f'Downloading default GADM data to {default_gadm_path}')
        download_gadm(default_gadm_path)
        verify_gadm(default_gadm_path)
        print(f'Using default GADM data from {default_gadm_path}')


_sf = None
def get_gadm():
    """Get the country shapefile"""
    if _sf is None:
        self.print(f'Loading country shapefile...')
        _sf = gpd.read_file(f'{dirname(abspath(__file__))}/gadm_0/gadm36_0.shp')
    return self._sf


if __name__ == '__main__':
    #DEBUG
    download_gadm(default_gadm_path)