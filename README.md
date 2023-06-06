# Notes

## Selecting Data
🤷

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