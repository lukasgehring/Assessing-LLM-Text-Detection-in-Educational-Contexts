#!/bin/bash

echo "Loading results..."
rsync -avP citec-cluster:projects/BenchEduLLMDetect/results/ .

echo "Loading logs..."
rsync -avP citec-cluster:projects/BenchEduLLMDetect/logs/ ../logs/