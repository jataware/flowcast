from __future__ import annotations
from itertools import product

import pandas as pd
import csv
from io import StringIO
from h3 import edge_length, geo_to_h3
from tqdm import tqdm
from pathlib import Path
from dataclasses import dataclass, field
from abc import ABC
import xarray as xr
import numpy as np
from typing import Literal
from statistics import mode

##
import pdb
from matplotlib import pyplot as plt

# some known values
n_rows = 5164970  # not including header
chunk_size = 50000
n_chunks = n_rows // chunk_size + 1
h3_resolution = 7
h3_edge_length = edge_length(h3_resolution, 'km')
neighbor_distance = np.sqrt(3) * h3_edge_length
resolution_multiplier = 1
resolution_degrees = neighbor_distance / 111.32 * resolution_multiplier

us_lat = np.arange(20+resolution_degrees/2, 90, resolution_degrees)
us_lon = np.arange(0+resolution_degrees/2, 360, resolution_degrees)
au_lat = np.arange(-40+resolution_degrees/2, -10, resolution_degrees)
au_lon = np.arange(0+resolution_degrees/2, 360, resolution_degrees)

# lat divider:
# us > 0 degrees
# au < 0 degrees

whole_df = None


def load_whole_df():
    global whole_df
    print('loading whole df into memory...', end='', flush=True)
    whole_df = pd.read_csv('reprocessed_datacube.csv')
    print('done')

    # rename the columns and drop the ones we don't need
    whole_df = process_chunk(whole_df)


def process_chunk(chunk):
    """Process each chunk of the DataFrame."""
    cols_to_keep = [
        'H3_Address',
        # 'H3_Resolution',
        # 'H3_Geometry',
        'Longitude_EPSG4326',
        'Latitude_EPSG4326',
        # 'Continent_Majority',
        # 'Continent_Minority',
        # 'Country_Majority',
        # 'Country_Minority',
        # 'Province_Majority',
        # 'Province_Minority',
        # 'Terrane_Majority',
        # 'Terrane_Minority',
        # 'Terrane_Contact',
        # 'Terrane_Proximity',
        'Geology_Eon_Maximum_Majority',
        'Geology_Eon_Maximum_Minority',
        'Geology_Eon_Minimum_Majority',
        'Geology_Eon_Minimum_Minority',
        'Geology_Era_Maximum_Majority',
        'Geology_Era_Maximum_Minority',
        'Geology_Era_Minimum_Majority',
        'Geology_Era_Minimum_Minority',
        'Geology_Period_Maximum_Majority',
        'Geology_Period_Maximum_Minority',
        'Geology_Period_Minimum_Majority',
        'Geology_Period_Minimum_Minority',
        'Geology_Period_Contact',
        'Geology_Lithology_Majority',
        'Geology_Lithology_Minority',
        'Geology_Lithology_Contact',
        'Geology_Dictionary_Alkalic',
        'Geology_Dictionary_Anatectic',
        'Geology_Dictionary_Calcareous',
        'Geology_Dictionary_Carbonaceous',
        'Geology_Dictionary_Cherty',
        'Geology_Dictionary_CoarseClastic',
        'Geology_Dictionary_Evaporitic',
        'Geology_Dictionary_Felsic',
        'Geology_Dictionary_FineClastic',
        'Geology_Dictionary_Gneissose',
        'Geology_Dictionary_Igneous',
        'Geology_Dictionary_Intermediate',
        'Geology_Dictionary_Pegmatitic',
        'Geology_Dictionary_RedBed',
        'Geology_Dictionary_Schistose',
        'Geology_Dictionary_Sedimentary',
        'Geology_Dictionary_UltramaficMafic',
        'Geology_PassiveMargin_Proximity',
        'Geology_BlackShale_Proximity',
        'Geology_Fault_Proximity',
        'Geology_CoverThickness',
        'Geology_Paleolongitude_Period_Maximum',
        'Geology_Paleolongitude_Period_Minimum',
        'Geology_Paleolatitude_Period_Maximum',
        'Geology_Paleolatitude_Period_Minimum',
        'Seismic_LAB_Hoggard',
        'Seismic_LAB_Priestley',
        'Seismic_Moho',
        'Seismic_Moho_GEMMA',
        'Seismic_Moho_Szwillus',
        'Seismic_Velocity_050km',
        'Seismic_Velocity_100km',
        'Seismic_Velocity_150km',
        'Seismic_Velocity_200km',
        'Gravity_GOCE_Differential',
        'Gravity_GOCE_MaximumCurve',
        'Gravity_GOCE_MinimumCurve',
        'Gravity_GOCE_MeanCurve',
        'Gravity_GOCE_ShapeIndex',
        'Gravity_Bouguer',
        'Gravity_Bouguer_BGI',
        'Gravity_Bouguer_HGM',
        'Gravity_Bouguer_HGM_Worms_Proximity',
        'Gravity_Bouguer_UpCont30km',
        'Gravity_Bouguer_UpCont30km_HGM',
        'Gravity_Bouguer_UpCont30km_HGM_Worms_Proximity',
        'Magnetic_RTP',
        'Magnetic_EMAG2v3',
        'Magnetic_EMAG2v3_CuriePoint',
        'Magnetic_1VD',
        'Magnetic_HGM',
        'Magnetic_HGM_Worms_Proximity',
        'Magnetic_LongWavelength_HGM',
        'Magnetic_LongWavelength_HGM_Worms_Proximity',
        'HeatFlow',
        'Magnetotelluric',
        'Litmod_Density_Asthenosphere',
        'Litmod_Density_Crust',
        'Litmod_Density_Lithosphere',
        'Crust1_Type',
        'Crust1_CrustalThickness',
        'Crust1_SedimentThickness',
        'Training_MVT_Deposit',
        'Training_MVT_Occurrence',
        'Training_CD_Deposit',
        'Training_CD_Occurrence',
    ]

    # Drop columns not in cols_to_keep
    chunk = chunk[cols_to_keep]

    # rename longitude and latitude columns
    chunk = chunk.rename(columns={'Longitude_EPSG4326': 'longitude',
                                  'Latitude_EPSG4326': 'latitude'})

    return chunk


def fix_encoding():
    input_file = Path('2021_Table04_Datacube.csv')
    output_file = Path('reprocessed_datacube.csv')

    # Define chunk size
    chunk_size = 1000000  # Adjust this based on your memory constraints

    number_of_lines = sum(1 for line in open(input_file, 'rb'))

    data = []
    if output_file.exists():
        output_file.unlink()
    with open(input_file, 'rb') as file:
        for i, line in tqdm(enumerate(file), total=number_of_lines, desc='cleaning csv'):
            try:
                # Decode with UTF-8
                decoded_line = line.decode('utf-8-sig')
            except UnicodeDecodeError:
                # If it fails, try another encoding
                decoded_line = line.decode('latin1')

            # Use csv.reader to handle complex CSV parsing
            reader = csv.reader(StringIO(decoded_line))
            for parsed_line in reader:
                data.append(parsed_line)

            if (i + 1) % chunk_size == 0:
                # convert to dataframe
                df = pd.DataFrame(data)

                # append to csv
                df.to_csv(output_file, mode='a', index=False,
                          header=False, encoding='utf-8')
                data = []

        # convert remaining to dataframe and append
        df = pd.DataFrame(data)
        df.to_csv(output_file, mode='a', index=False,
                  header=False, encoding='utf-8')


class Span(ABC):
    """Span of data in a column. Could be numeric range, list of strings, etc."""

    def update(self, value):
        raise NotImplementedError

    def batch_update(self, values):
        raise NotImplementedError


@dataclass
class NumericSpan(Span):
    """Numeric span of data in a column."""
    min: float = float('inf')
    max: float = float('-inf')

    def update(self, value: float):
        if value < self.min:
            self.min = value
        if value > self.max:
            self.max = value

    def batch_update(self, values: pd.Series):
        self.min = min(self.min, values.min())
        self.max = max(self.max, values.max())


@dataclass
class StringSpan(Span):
    """String span of data in a column."""
    values: set[str] = field(default_factory=set)

    def update(self, value: str):
        self.values.add(value)

    def batch_update(self, values: pd.Series):
        self.values.update(values.unique())


# @dataclass
# class BooleanSpan(Span):
#     """Boolean span of data in a column."""
#     values: set[bool] = field(default_factory=set)

#     def update(self, value: bool):
#         self.values.add(value)

#     def batch_update(self, values: pd.Series):
#         self.values.update(values.unique())


def determine_span_type(column: pd.Series) -> Span:
    """Determine the span type of a column."""
    if column.dtype == 'float64' or column.dtype == 'int64':
        return NumericSpan()
    elif column.dtype == 'object':
        return StringSpan()
    # elif column.dtype == 'bool':
    #     return BooleanSpan()
    else:
        raise TypeError(f'Unknown column type: {column.dtype}')


def get_column_values(data_path, column_name, chunk_size, leave=False):

    # short circuit if we have the whole df
    if whole_df is not None:
        return whole_df[column_name].values

    df_chunks = pd.read_csv(data_path, chunksize=chunk_size)

    col_chunks = []
    for raw_chunk in tqdm(df_chunks, total=n_chunks, desc=f'grabbing {column_name}', leave=leave):
        chunk = process_chunk(raw_chunk)
        col_chunks.append(chunk[column_name])

    return pd.concat(col_chunks).values


def get_spans_chunked(input_file):
    df_chunks = pd.read_csv(input_file, chunksize=chunk_size)
    col_names = None

    chunk0 = process_chunk(next(df_chunks))
    col_names = chunk0.columns

    spans: dict[str, Span] = {}
    for chunk in tqdm(df_chunks, total=n_chunks, desc='determining spans'):
        chunk = process_chunk(chunk)

        for col_name in col_names:
            series = chunk[col_name]
            if col_name not in spans:
                spans[col_name] = determine_span_type(series)
            spans[col_name].batch_update(series)

    return spans


def get_spans(df: pd.DataFrame):
    col_names = df.columns
    spans: dict[str, Span] = {}
    for col_name in tqdm(col_names, total=len(col_names), desc='determining spans'):
        series = df[col_name]
        spans[col_name] = determine_span_type(series)
        spans[col_name].batch_update(series)

    return spans


def create_dataarray():
    input_file = Path('reprocessed_datacube.csv')

    h3_addresses = get_column_values(input_file, 'H3_Address', chunk_size, leave=True)

    us_dataset_dict = {}
    us_bucket_map, us_lat_min_idx, us_lat_max_idx, us_lon_min_idx, us_max_lon_idx = create_bucket_map(
        h3_addresses, us_lat, us_lon)

    au_dataset_dict = {}
    au_bucket_map, au_lat_min_idx, au_lat_max_idx, au_lon_min_idx, au_max_lon_idx = create_bucket_map(
        h3_addresses, au_lat, au_lon)

    # determine the span of data in each of the columns
    if whole_df is not None:
        spans = get_spans(whole_df)
    else:
        spans = get_spans_chunked(input_file)

    # create the data array for each feature
    for col_name, span in tqdm(spans.items(), total=len(spans), desc='creating dataset'):
        if col_name.lower() in ['longitude', 'latitude', 'h3_address']:
            continue

        col_data = get_column_values(input_file, col_name, chunk_size)
        if isinstance(span, NumericSpan):
            us_feature = gridify(us_bucket_map, us_lat, us_lon, col_data, col_name)
            us_feature = us_feature[us_lat_min_idx:us_lat_max_idx+1, us_lon_min_idx:us_max_lon_idx+1]
            us_dataset_dict[col_name] = (['lat', 'lon'], us_feature)

            au_feature = gridify(au_bucket_map, au_lat, au_lon, col_data, col_name)
            au_feature = au_feature[au_lat_min_idx:au_lat_max_idx+1, au_lon_min_idx:au_max_lon_idx+1]
            au_dataset_dict[col_name] = (['lat', 'lon'], au_feature)

        elif isinstance(span, StringSpan):
            # map each string to an int
            non_nan_span = [s for s in span.values if not pd.isna(s)]
            non_nan_span = {s: i for i, s in enumerate(non_nan_span)}
            col_data = np.array([non_nan_span[s] if not pd.isna(s) else np.nan for s in col_data], dtype=float)

            us_feature = gridify(us_bucket_map, us_lat, us_lon, col_data, col_name)
            us_feature = us_feature[us_lat_min_idx:us_lat_max_idx+1, us_lon_min_idx:us_max_lon_idx+1]
            us_dataset_dict[col_name] = (['lat', 'lon'], us_feature, {'legend': [*non_nan_span]})

            au_feature = gridify(au_bucket_map, au_lat, au_lon, col_data, col_name)
            au_feature = au_feature[au_lat_min_idx:au_lat_max_idx+1, au_lon_min_idx:au_max_lon_idx+1]
            au_dataset_dict[col_name] = (['lat', 'lon'], au_feature, {'legend': [*non_nan_span]})
        else:
            pdb.set_trace()
            raise TypeError(f'Unknown span type: {type(span)}')

    # create dataset and save
    print('saving dataset...', end='', flush=True)

    us_dataset = xr.Dataset(us_dataset_dict, coords={
                            'lat': us_lat[us_lat_min_idx:us_lat_max_idx+1], 'lon': us_lon[us_lon_min_idx:us_max_lon_idx+1]})
    us_dataset.to_netcdf('us_datacube.nc')

    au_dataset = xr.Dataset(au_dataset_dict, coords={
                            'lat': au_lat[au_lat_min_idx:au_lat_max_idx+1], 'lon': au_lon[au_lon_min_idx:au_max_lon_idx+1]})
    au_dataset.to_netcdf('au_datacube.nc')

    print('done')


def create_bucket_map(h3_addresses: np.ndarray, output_lat: np.ndarray, output_lon: np.ndarray) -> list[list[str] | None]:
    mapping: dict[tuple[int, int], list[int]] = {}

    # convert h3_addresses to dictionary
    h3_addresses: dict[str, int] = {h3_address: i for i, h3_address in enumerate(h3_addresses)}

    max_lat_idx = None
    min_lat_idx = None
    max_lon_idx = None
    min_lon_idx = None

    for (lat_idx, lat), (lon_idx, lon) in tqdm(product(
        enumerate(output_lat),
        enumerate(output_lon)
    ), total=len(output_lat)*len(output_lon), desc='creating bucket map'):
        h3_address = geo_to_h3(lat, lon, h3_resolution)
        data_idx = h3_addresses.get(h3_address, None)
        if data_idx is None:
            continue
        mapping[(lat_idx, lon_idx)] = data_idx

        # keep track of the min/max lat/lon indices
        if max_lat_idx is None or lat_idx > max_lat_idx:
            max_lat_idx = lat_idx
        if min_lat_idx is None or lat_idx < min_lat_idx:
            min_lat_idx = lat_idx
        if max_lon_idx is None or lon_idx > max_lon_idx:
            max_lon_idx = lon_idx
        if min_lon_idx is None or lon_idx < min_lon_idx:
            min_lon_idx = lon_idx

    return mapping, min_lat_idx, max_lat_idx, min_lon_idx, max_lon_idx


def gridify(
    bucket_map: dict[tuple[int, int], int],
    output_lat: np.ndarray,
    output_lon: np.ndarray,
    data: np.ndarray,
    name: str,
) -> np.ndarray:
    rows = len(output_lat)
    cols = len(output_lon)
    grid = np.full((rows, cols), np.nan, dtype=np.float32)

    for (lat_idx, lon_idx), index in tqdm(bucket_map.items(), total=len(bucket_map), desc=f'gridifying {name}', leave=False):
        try:
            grid[lat_idx, lon_idx] = data[index]
        except IndexError:
            pdb.set_trace()
            ...

    return grid


"""
#creating the data array with categorical features as ints:
import numpy as np
import xarray as xr

category_names = {0:'category A', 1:'category B', 2:'category C'}

latitudes = np.linspace(-90, 90, 10)  # 10 latitude points
longitudes = np.linspace(-180, 180, 10)  # 10 longitude points

# Create a 2D grid of categorical data
grid_data = np.random.randint(0, 3, size=(latitudes.size, longitudes.size))

# Create the xarray Dataset with attributes
ds_grid = xr.Dataset({
    'categorical_feature': (['lat', 'lon'], grid_data, {'category_legend': category_names})
}, coords={
    'lat': latitudes,
    'lon': longitudes
}) 
"""


if __name__ == '__main__':
    if not Path('reprocessed_datacube.csv').exists():
        fix_encoding()

    try:
        load_whole_df()
    except:
        print('failed to load whole df. proceeding in chunked mode.')

    create_dataarray()
