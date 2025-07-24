import os
import re

log_files = []
for _, _, files in os.walk('.'):
    for file in files:
        if file.endswith('.log'):
            log_files.append(file)

for path, _, files in os.walk('../results'):
    for file in files:
        if file.endswith('.log'):
            job_id = (path.split('/')[-1])
            log_files = [log_file for log_file in log_files if job_id not in log_file]

print("not used log files", log_files)

#for file in log_files:
#    os.remove(f"../logs/{file}")