FROM ubuntu:latest

# Install required system packages
RUN apt-get update -y && \
    apt-get install -y wget bzip2


# Install required system packages
RUN apt-get update -y && \
    apt-get install -y wget bzip2

# Install Anaconda
RUN wget https://repo.anaconda.com/archive/Anaconda3-2023.03-1-Linux-x86_64.sh && \
    bash Anaconda3-2023.03-1-Linux-x86_64.sh -b -p /opt/anaconda && \
    rm Anaconda3-2023.03-1-Linux-x86_64.sh

# Set path to conda
ENV PATH /opt/anaconda/bin:$PATH

# Updating Conda packages
# RUN conda update conda -y && conda update --all -y

# Create conda environment
RUN conda create -n cmip6 python=3.10 -y

# Make RUN commands use the new environment
SHELL ["conda", "run", "-n", "cmip6", "/bin/bash", "-c"]


# Add your library to the Docker image
COPY . /app

# Set the working directory
WORKDIR /app

# install other required packages
RUN apt-get install -y libexpat1
RUN conda install -c conda-forge numpy=1.24.4 -y
RUN conda install -c nesii -c conda-forge esmpy=8.4.2 -y
RUN pip install -r requirements.txt


# set up entrypoint without providing the command line argument
# CMD ["/bin/bash", "-c", "conda run -n cmip6 python ./test.py ssp585"]


# Create entrypoint.sh, make it executable and set it as ENTRYPOINT
RUN echo '#!/bin/bash' > /entrypoint.sh && \
    echo 'source /opt/anaconda/etc/profile.d/conda.sh' >> /entrypoint.sh && \
    echo 'conda activate cmip6' >> /entrypoint.sh && \
    echo 'exec python "./test.py" "$@"' >> /entrypoint.sh && \
    chmod +x /entrypoint.sh

ENTRYPOINT ["/entrypoint.sh"]