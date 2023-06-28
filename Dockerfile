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
# RUN conda update conda -y 
#  && conda update --all -y

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



######### Install CDO from source #########

# install clang from anaconda
# RUN conda install -c anaconda clang

# # download CDO source code
# RUN wget https://code.mpimet.mpg.de/attachments/download/28013/cdo-2.2.0.tar.gz

# # extract CDO source code
# RUN tar -xvf cdo-2.2.0.tar.gz

# change directory to CDO source code


# RUN conda install -c conda-forge cdo
# RUN conda install -c conda-forge python-cdo




# Install your library's Python dependencies
# RUN pip install -r requirements.txt

# Define the command to run your application
CMD ["/bin/bash", "-c", "conda run -n cmip6 python ./test.py ssp585"]
# CMD ["python", "./test2.py"]
# CMD ["python", "./test.py"]
