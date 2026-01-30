#!/bin/bash

mapfile -t PARAMS < <(
    cat \
      params/aae/roberta_local.txt \
      params/bawe/roberta_local.txt \
      params/persuade/roberta_local.txt \
    | grep -vE '^\s*$'
)

cd ../ || exit

for param in "${PARAMS[@]}"; do
  echo "====================================="
  echo "Parameter: ${param}"
  echo "====================================="
  python -u main.py "$@" ${param}
done
