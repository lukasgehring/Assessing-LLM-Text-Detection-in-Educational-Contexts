#!/bin/bash

mapfile -t PARAMS < <(cat params/aae/ghostbuster.txt params/bawe/ghostbuster.txt params/persuade/ghostbuster.txt | grep -vE '^\s*$')

cd ../ || exit


while true; do
    read -p "Please enter your OpenAI API key: " OPENAI_API_KEY

    if [[ -n "$OPENAI_API_KEY" ]]; then
        break
    else
        echo "API key cannot be empty. Please try again."
    fi
done

for param in "${PARAMS[@]}"; do
  echo "====================================="
  echo "Parameter: ${param}"
  echo "====================================="
  python -u main.py "$@" ${param} --openai_key "$OPENAI_API_KEY"
done
