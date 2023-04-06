FROM ubuntu:22.04

ENV LANG=C.UTF-8 LC_ALL=C.UTF-8
ENV PATH /opt/conda/bin:$PATH

RUN apt-get update && \
    apt-get -y upgrade
RUN apt-get install -y curl tini

RUN apt-get clean

RUN /usr/sbin/groupadd -g 1000 user && \
    /usr/sbin/useradd -u 1000 -g 1000 -d /opt/redmapper redmapper && \
    mkdir /opt/redmapper && chown redmapper.user /opt/redmapper && \
    chown -R redmapper.user /opt

COPY . /opt/redmapper/workdir
RUN chown -R redmapper.user /opt/redmapper/workdir

USER redmapper

RUN curl -L -o ~/mambaforge.sh https://github.com/conda-forge/miniforge/releases/download/23.1.0-1/Mambaforge-23.1.0-1-Linux-x86_64.sh && \
    /bin/bash ~/mambaforge.sh -b -p /opt/conda && \
    rm ~/mambaforge.sh
RUN . /opt/conda/etc/profile.d/conda.sh && conda create --yes --name redmapper-env
RUN echo ". /opt/conda/etc/profile.d/conda.sh" > /opt/redmapper/startup.sh && \
    echo "conda activate redmapper-env" >> /opt/redmapper/startup.sh

RUN . /opt/conda/etc/profile.d/conda.sh && conda activate redmapper-env && \
    mamba install --yes python=3.10 numpy scipy astropy matplotlib pyyaml gsl c-compiler fitsio esutil healpy healsparse hpgeom && \
    conda clean -af --yes

RUN . /opt/conda/etc/profile.d/conda.sh && conda activate redmapper-env && \
    cd /opt/redmapper/workdir && \
    python setup.py install

ENTRYPOINT [ "/usr/bin/tini", "--" ]
CMD [ "/bin/bash", "-lc" ]
