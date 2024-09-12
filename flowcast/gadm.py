
import os
import requests
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm
import geofeather as gf
from geopandas import GeoDataFrame


from functools import lru_cache
cache = lru_cache(maxsize=None)


import pdb


default_gadm_path = Path(__file__).parent/'gadm'
gadm_path = None
gadm_env_var = 'GADM_DIR'


GADM_FILES = [
    'gadm36_0.feather',
    'gadm36_2.feather',
    'gadm36_3.feather'
]

def verify_gadm_dir(dir:Path):
    """Presuming that GADM data is already downloaded, verify that the given directory contains the GADM data in the expected format"""
    #verify that the directory exists
    assert dir.is_dir(), f'Invalid GADM directory. Expected a directory. Got: {dir}'

    for filename in GADM_FILES:
        verify_gadm_file(dir/filename)


def verify_gadm_file(file:Path):
    """Verify that the given file is a valid GADM file"""
    level = file.stem.split('_')[-1]
    assert file.is_file(), f'Invalid GADM directory. No admin{level} data ({file.name}) found in {file.parent}'
    gf.from_geofeather(file)


def download_gadm(dir:Path):
    """Download GADM data to the given directory"""

    # create the directory if it doesn't exist
    os.makedirs(dir, exist_ok=True)

    # download each file
    root = 'https://jataware-world-modelers.s3.amazonaws.com/gadm/'
    for filename in GADM_FILES:
        url = f'{root}{filename}.zip'
        download_gadm_file(url, filename, dir)


def download_gadm_file(url:str, filename:str, dir:Path):
    """Download a GADM file from the S3 bucket"""
    # check if the file already exists
    if (dir/filename).is_file():
        print(f'{filename} already exists. Skipping download')
        return

    # add the .zip extension to the filename
    assert filename.endswith('.feather'), f'Invalid filename. Expected a .feather file. Got: {filename}'
    filename = f'{filename}.zip'

    # download the file
    r = requests.get(url, allow_redirects=True, stream=True)
    assert r.status_code == 200, f'Failed to download GADM data from {url}. Got status code: {r.status_code}'

    # get the total size of the file for the progress bar
    total_size = int(r.headers.get('content-length', 0))

    # save the file
    with open(dir/filename, 'wb') as f, tqdm(desc=f'download {filename}', total=total_size, unit='iB', unit_scale=True) as bar:
        for chunk in r.iter_content(chunk_size=1024):
            f.write(chunk)
            bar.update(len(chunk))

    # unzip the file
    unzip_gadm_file(filename, dir)


def unzip_gadm_file(filename:str, dir:Path):
    """Unzip a GADM file with a progress bar into the given directory"""

    try:
        with ZipFile(dir/filename, 'r') as z:
            # Get the total uncompressed size for the progress bar
            total_size = sum(item.file_size for item in z.infolist())
            with tqdm(desc=f'unzip {filename}', total=total_size, unit='B', unit_scale=True, unit_divisor=1024) as bar:
                
                for member in z.infolist():
                    # Extract each member in chunks to update the progress bar
                    with z.open(member, 'r') as source, open(dir/member.filename, 'wb') as target:
                        chunk_size = 1024 * 1024  # 1MB
                        while True:
                            chunk = source.read(chunk_size)
                            if not chunk:
                                break
                            target.write(chunk)
                            bar.update(len(chunk))

    # delete the zip file
    finally:
        os.remove(dir/filename)

def setup_gadm():
    """Ensure that GADM data is downloaded and ready to use"""
    global gadm_path
    # if GADM_DIR environment variable is set, use that
    _gadm_path = os.environ.get(gadm_env_var)
    if _gadm_path is not None:
        gadm_path = Path(_gadm_path)
        verify_gadm_dir(gadm_path)
        print(f'Using ENV specified GADM data from {gadm_path}')
        return

    # otherwise use the default directory, and download GADM if necessary
    gadm_path = default_gadm_path
    try:
        verify_gadm_dir(gadm_path)
        print(f'Using default GADM data from {gadm_path}')
        return
    except:
        print(f'Downloading default GADM data to {gadm_path}')
        download_gadm(gadm_path)
        verify_gadm_dir(gadm_path)
        print(f'Using default GADM data from {gadm_path}')

@cache
def get_admin0_shapes():
    assert gadm_path is not None, f'GADM data not initialized. Call setup_gadm() first'
    shapes = gf.from_geofeather(gadm_path/'gadm36_0.feather')
    return shapes

# @cache
# def get_admin1_shapes():
#     raise NotImplementedError("TODO: aggregate admin1 shapes from admin2 shapes")

@cache
def get_admin2_shapes() -> GeoDataFrame:
    assert gadm_path is not None, f'GADM data not initialized. Call setup_gadm() first'
    shapes = gf.from_geofeather(gadm_path/'gadm36_2.feather')
    return shapes

@cache
def get_admin3_shapes() -> GeoDataFrame:
    assert gadm_path is not None, f'GADM data not initialized. Call setup_gadm() first'
    shapes = gf.from_geofeather(gadm_path/'gadm36_3.feather')
    return shapes

# _sf = None
# def get_gadm():
#     """Get the country shapefile"""
#     if _sf is None:
#         self.print(f'Loading country shapefile...')
#         _sf = gpd.read_file(f'{dirname(abspath(__file__))}/gadm_0/gadm36_0.shp')
#     return self._sf


if __name__ == '__main__':
    #DEBUG
    # download_gadm(default_gadm_path)
    # get_admin2_shapes()
    get_admin3_shapes()