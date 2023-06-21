# Notes

## Selecting CMIP6 Data
- browse data at https://esgf-node.llnl.gov/search/cmip6/
- 🤷 unclear how to determine what the correct variable name is for some desired type of measurement
- also unclear best way to pick a dataset e.g. there are > 55,000 entries for "tasmax" (maximum surface air temperature)
- for projected future data, a good idea is to set the experiment id to one of the SSPs (shared socioeconomic pathways) e.g. ssp585 (high emissions) or ssp126 (low emissions)
### Population Data
- https://sedac.ciesin.columbia.edu/data/set/popdynamics-1-8th-pop-base-year-projection-ssp-2000-2100-rev01

## Downloading Data
### Prereqs
- make an account: https://esgf-node.llnl.gov/user/add/?next=http://esgf-node.llnl.gov/search/cmip6/
- grab OpenID for your account, e.g. https://esgf-node.llnl.gov/esgf-idp/openid/your_username

### Downloading
- Identify target data at: https://esgf-node.llnl.gov/search/cmip6/
- download the WGET script for the data
- make script executable, e.g. 
    ```bash
    chmod +x wget-20230606122309.sh
    ```
- run script e.g. 
    ```bash
    ./wget-20230606122309.sh -s # -s should allow downloading without needing openID login
    ```
- enter OpenID if prompted
- enter password if prompted
- data will download to current directory


## Setup
```bash
conda install -c conda-forge xesmf
pip install dask
pip install nc-time-axis
```


## Extreme Heat Scenario
**How many people will be exposed to extreme heat events (e.g., heatwaves) in the future?**

1. download (geospatial) decadal population data for ssp585 (high emissions) (2010-2100)
1. interpolate population data to yearly
1. download (geospatial) monthly maximum surface air temperature (tasmax) projections for ssp585 (2015-2100)
1. threshold tasmax by 35°C to identify (geospatial) extreme heat months
1. aggregate extreme heat months by year to get years containing any extreme heat events
1. geospatial bilinear interpolation to regrid extreme heat data to match population data grid
1. multiply extreme heat data by population data to get (geospatial) population exposed to extreme heat by year
1. clip data by each country's shapefile to get (geospatial) population exposed to extreme heat by year by country
1. sum over area of each country to get total population exposed to extreme heat by year by country
1. view results


## Crop Viability Scenario
**How will the viability of current croplands change in the future?**
### Process

1. Get (geospatial) land use data. Modis data just includes africa. Extract points (mask) corresponding to cropland
1. Get (geospatial) temperature (tas) and precipitation (pr) data projected from 2015-2100
1. Regrid tas and pr to match cropland data resolution, and select masked values
1. Compute distribution (mean/std) of tas and pr from 2015-2020 to use as baseline climate for croplands
1. Compute the z-score of tas and pr over all time (2015-2100)
1. Viability is any location where |tas_z| < 3.0 and |pr_z| < 3.0. Compute the viability % over time as compared to the baseline
1. Sum viability over lat and lon to get single time curve
1. View results
