import subprocess
import os
import time

while os.path.exists("/proc/253683"):
    print("GPT2 is training....")
    time.sleep(60*15)

subprocess.call(['sh', './run.sh'])
