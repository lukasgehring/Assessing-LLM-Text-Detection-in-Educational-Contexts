#!/bin/bash

echo "Loading results..."
rsync -avP citec-cluster:projects/BenchEduLLMDetect/results/ . --dry-run

echo "Loading logs..."
rsync -avP citec-cluster:projects/BenchEduLLMDetect/logs/ ../logs/ --dry-run
