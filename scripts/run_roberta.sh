#!/bin/bash

mapfile -t PARAMS < <(cat params/aae/roberta.txt params/bawe/roberta.txt params/persuade/roberta.txt | grep -vE '^\s*$')

cd ../ || exit

for param in "${PARAMS[@]}"; do
  echo "====================================="
  echo "Parameter: ${param}"
  echo "====================================="
  python -u main.py "$@" ${param}
done
