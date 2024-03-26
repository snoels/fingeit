import subprocess


request_command = """gcloud compute instances create supergeldgeit \
    --project=silverfin-216312 \
    --zone={zone} \
    --machine-type=a2-highgpu-1g \
    --network-interface=network-tier=PREMIUM,stack-type=IPV4_ONLY,subnet=default \
    --maintenance-policy=TERMINATE \
    --provisioning-model=STANDARD \
    --service-account=403967595445-compute@developer.gserviceaccount.com \
    --scopes=https://www.googleapis.com/auth/devstorage.read_only,https://www.googleapis.com/auth/logging.write,https://www.googleapis.com/auth/monitoring.write,https://www.googleapis.com/auth/servicecontrol,https://www.googleapis.com/auth/service.management.readonly,https://www.googleapis.com/auth/trace.append \
    --accelerator=count=1,type=nvidia-tesla-a100 \
    --create-disk=auto-delete=yes,boot=yes,device-name=supergeldgeit,image=projects/ubuntu-os-cloud/global/images/ubuntu-2204-jammy-v20240228,mode=rw,size=50,type=projects/silverfin-216312/zones/us-central1-b/diskTypes/pd-balanced \
    --no-shielded-secure-boot \
    --shielded-vtpm \
    --shielded-integrity-monitoring \
    --labels=goog-ec-src=vm_add-gcloud \
    --reservation-affinity=any
"""
a_100_available_zones = [
    "asia-northeast1-a",
    "asia-northeast1-c",
    "asia-northeast3-a",
    "asia-northeast3-b",
    "asia-southeast1-b",
    "asia-southeast1-c",
    "europe-west4-a",
    "europe-west4-b",
    "me-west1-b",
    "me-west1-c",
    "us-central1-a",
    "us-central1-b",
    "us-central1-c",
    "us-central1-f",
    "us-east1-b",
    "us-west1-b",
    "us-west3-b",
    "us-west4-b"
]


for zone in a_100_available_zones:
    zoned_request_command = request_command.format(zone=zone)
    process = subprocess.Popen(zoned_request_command , shell=True, stdout=subprocess.PIPE)
    process.wait()
    stdout, stderr = process.communicate()
      
    print(process.returncode)
    if process.returncode == 0:
        print("===================================\nWoop woop spinning up supergeldgeit\n===================================")
        break
    else:
        print("===================================\nNo luck, continuing the search\n===================================")
