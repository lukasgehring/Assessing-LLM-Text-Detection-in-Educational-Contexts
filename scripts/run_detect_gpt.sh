#!/bin/bash

mapfile -t PARAMS < <(cat params/aae/detect_gpt.txt params/bawe/detect_gpt.txt params/persuade/detect_gpt.txt | grep -vE '^\s*$')

cd ../ || exit

for param in "${PARAMS[@]}"; do
  echo "====================================="
  echo "Parameter: ${param}"
  echo "====================================="
  python -u main.py "$@" ${param}
done
