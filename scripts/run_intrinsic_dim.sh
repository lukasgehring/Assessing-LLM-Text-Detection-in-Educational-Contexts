#!/bin/bash

mapfile -t PARAMS < <(cat params/aae/intrinsic_dim.txt params/bawe/intrinsic_dim.txt params/persuade/intrinsic_dim.txt | grep -vE '^\s*$')

cd ../ || exit

for param in "${PARAMS[@]}"; do
  echo "====================================="
  echo "Parameter: ${param}"
  echo "====================================="
  python -u main.py "$@" ${param}
done
