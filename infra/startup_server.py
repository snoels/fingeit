""" Script to start the VM instance supergeldgeit in GCP.
        
This script starts the supergeldgeit instance in the asia-northeast1-c zone. 
The script will keep retrying to start the instance until it is successfully started. 
The external IP of the instance is printed to the console once the instance is started.
"""

import os
import re
import subprocess
from time import sleep


class color:
   PURPLE = '\033[95m'
   CYAN = '\033[96m'
   DARKCYAN = '\033[36m'
   BLUE = '\033[94m'
   GREEN = '\033[92m'
   YELLOW = '\033[93m'
   RED = '\033[91m'
   BOLD = '\033[1m'
   UNDERLINE = '\033[4m'
   END = '\033[0m'


def get_external_ip():
    try:
        ask_vm_info = "gcloud compute instances describe supergeldgeit --zone=asia-northeast1-c"
        process = subprocess.Popen(ask_vm_info , shell=True, stdout=subprocess.PIPE)
        process.wait()
        stdout, stderr = process.communicate()
        return re.search(r"natIP: ([1-9]*.){4}",str(stdout)).group(0).split(' ')[1]
    except Exception as e:
        print(e)
   

while True:
    command = "gcloud compute instances start supergeldgeit --zone=asia-northeast1-c"
    process = subprocess.Popen(command , shell=True, stdout=subprocess.PIPE)
    process.wait()
   
    if process.returncode == 0:
        ip = get_external_ip()
        print(f"===================================\n{color.GREEN}Starting supergeldgeit.{color.END}\n===================================\n {color.BLUE}External IP: {ip}{color.END}\n===================================")
        break
    else:
        print(f"===================================\n{color.RED}Retrying in 30s.{color.END}\n===================================")
        sleep(30)





