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
    ./wget-20230606122309.sh
    ```
- enter OpenID when prompted
- enter password when prompted
- data will download to current directory


## Setup
```bash
conda install -c conda-forge xesmf
```