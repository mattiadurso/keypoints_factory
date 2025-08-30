import os 
import yaml
from pathlib import Path


# wrapper to download files from a URL
wrappers_to_download = ['disk']

# load dict with links for download
with open("download_wrappers.yaml", "r") as f:
    wrappers_source = yaml.safe_load(f)

# just make sure method/ exists
os.makedirs("methods", exist_ok=True)

# download original implementations
for name in wrappers_source.keys():
    if name in wrappers_to_download or len(wrappers_to_download) == 0:
        for file_info in wrappers_source[name]:
            url  = file_info["url"]
            dest = file_info["dest"]

            if url:
                if 'github' in url:
                    os.system(f"git clone --recursive {url} methods/{dest}")
                else:
                    os.system(f"wget -P methods/{dest} {url}")

            if file_info.get("submodules"):
                for submodule in file_info["submodules"]:
                    os.system(submodule)
