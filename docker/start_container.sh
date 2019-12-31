#!/usr/bin/env bash

docker start test-container

docker exec test-container /bin/bash -c "/opt/conda/bin/jupyter notebook --notebook-dir=/opt/notebooks --ip='*' --port=8888 --no-browser --allow-root &"

docker exec test-container /bin/bash -c "sleep 5 && jupyter notebook list"
