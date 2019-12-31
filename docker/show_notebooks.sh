#!/usr/bin/env bash

docker exec test-container /bin/bash -c "sleep 5 && jupyter notebook list"
