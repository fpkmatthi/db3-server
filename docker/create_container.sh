#!/usr/bin/env bash

docker run -dit -p 8888:8888 --name test-container -v ~/Repositories/Hogent/Db3/rsrc:/opt/notebooks continuumio/anaconda3

docker exec test-container echo "Installing jupyter"

docker exec test-container /bin/bash -c "apt-get install -y dialog texlive-xetex texlive-fonts-recommended cm-super texlive-generic-recommended"

docker exec test-container /bin/bash -c "wget https://github.com/jgm/pandoc/releases/download/2.1.2/pandoc-2.1.2-1-amd64.deb"

docker exec test-container /bin/bash -c "dpkg -i pandoc-2.1.2-1-amd64.deb"

docker exec test-container /bin/bash -c "/opt/conda/bin/conda install jupyter -y --quiet && mkdir -p /opt/notebooks"

docker exec test-container echo "Launching notebooks"

docker exec test-container /bin/bash -c "/opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root &"

# docker attach test-container
