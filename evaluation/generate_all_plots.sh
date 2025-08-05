#!/bin/bash

target_dir_name="evaluation"

current_dir_name=$(basename "$PWD")

if [ "$current_dir_name" == "$target_dir_name" ]; then
    for file in *.py; do
        if [[ -f "$file" ]]; then
            echo "Running $file..."
            python "$file"
            if [[ $? -ne 0 ]]; then
                echo "Error occurred while running $file"
            fi
        fi
    done
else
    echo "Please run the script from: $target_dir_name"
fi
