#!/bin/bash

mapfile -t PARAMS < <(cat params/aae/fast_detect_gpt.txt params/bawe/fast_detect_gpt.txt params/persuade/fast_detect_gpt.txt | grep -vE '^\s*$')

cd ../ || exit

for param in "${PARAMS[@]}"; do
  echo "====================================="
  echo "Parameter: ${param}"
  echo "====================================="
  python -u main.py "$@" ${param}
done
