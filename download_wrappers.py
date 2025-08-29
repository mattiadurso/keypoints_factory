import os 
import yaml
from pathlib import Path


# wrapper to download files from a URL
wrappers_to_download = ['dedode']

# load dict with links for download
with open("download_wrappers.yaml", "r") as f:
    wrappers_source = yaml.safe_load(f)

# download original implementations
for name in wrappers_source.keys():
    if name in wrappers_to_download or len(wrappers_to_download) == 0:
        name_path = Path(name)
        if not name_path.exists():
            name_path.mkdir(parents=True, exist_ok=True)
        for file_info in wrappers_source[name]:
            url = file_info["url"]
            dest = file_info["dest"]

            if 'github' in url:
                os.system(f"git clone --recursive {url} wrappers/{name_path}/{dest}")
            else:
                os.system(f"wget -P wrappers/{name_path}/{dest} {url}")

            if file_info.get("submodules"):
                for submodule in file_info["submodules"]:
                    os.system(submodule)
