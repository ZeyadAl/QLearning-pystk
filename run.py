import os
import sys
import subprocess

#os.environ["GLOG_minloglevel"] = "2"

tracks = [
    "zengarden", 
    "lighthouse",
    "hacienda",
    "snowtuxpeak",
    "cornfield_crossing",
    "scotland"
]

for tr in tracks:
    os.system(f"python qlearning.py {tr} -n 2000") # for training
    cmd = ["python", "qlearning.py", str(tr), "--no-verbose"]

    with open("results.txt", "a") as out_f:
        # Redirect both stdout and stderr into the same file
        result = subprocess.run(
            cmd,
            stdout=out_f,
            stderr=subprocess.STDOUT,
            text=True
        )
