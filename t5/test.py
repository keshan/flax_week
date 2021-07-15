import subprocess
import os
import time

while os.path.exists("/proc/25870"):
    print("Roberta is training....")
    time.sleep(60*10)

subprocess.call(['sh', './run.sh'])
