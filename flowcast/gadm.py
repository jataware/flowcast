
import os
from os.path import dirname, abspath, isdir, isfile, join
import requests
from pathlib import Path
from zipfile import ZipFile
from tqdm import tqdm
import geofeather as gf
from geopandas import GeoDataFrame


from functools import lru_cache
cache = lru_cache(maxsize=None)


import pdb


default_gadm_path = Path(Path(__file__).parent, 'gadm')
gadm_path = None
gadm_env_var = 'GADM_DIR'



def verify_gadm(dir:Path):
    """Presuming that GADM data is already downloaded, verify that the given directory contains the GADM data in the expected format"""
    #verify that the directory exists
    assert isdir(dir), f'Invalid GADM directory. Expected a directory. Got: {dir}'
    assert isfile(join(dir, 'gadm36_2.feather')), f'Invalid GADM directory. No admin2 data (gadm36_2.feather) found in {dir}'
    assert isfile(join(dir, 'gadm36_3.feather')), f'Invalid GADM directory. No admin3 data (gadm36_3.feather) found in {dir}'

    #verify that the files are valid
    gf.from_geofeather(join(dir, 'gadm36_2.feather'))
    gf.from_geofeather(join(dir, 'gadm36_3.feather'))

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
    global gadm_path
    # if GADM_DIR environment variable is set, use that
    gadm_path = os.environ.get(gadm_env_var)
    if gadm_path is not None:
        gadm_path = Path(gadm_path)
        verify_gadm(gadm_path)
        print(f'Using ENV specified GADM data from {gadm_path}')
        return

    # otherwise use the default directory, and download GADM if necessary
    gadm_path = default_gadm_path
    try:
        verify_gadm(gadm_path)
        print(f'Using default GADM data from {gadm_path}')
        return
    except:
        print(f'Downloading default GADM data to {gadm_path}')
        download_gadm(gadm_path)
        verify_gadm(gadm_path)
        print(f'Using default GADM data from {gadm_path}')

# @cache
# def get_admin0_shapes():
#     admin2 = get_admin2_shapes()
#     pdb.set_trace()

# @cache
# def get_admin1_shapes():
#     raise NotImplementedError("TODO: aggregate admin1 shapes from admin2 shapes")

@cache
def get_admin2_shapes() -> GeoDataFrame:
    assert gadm_path is not None, f'GADM data not initialized. Call setup_gadm() first'
    shapes = gf.from_geofeather(join(gadm_path, 'gadm36_2.feather'))
    return shapes

@cache
def get_admin3_shapes() -> GeoDataFrame:
    assert gadm_path is not None, f'GADM data not initialized. Call setup_gadm() first'
    shapes = gf.from_geofeather(join(gadm_path, 'gadm36_3.feather'))
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