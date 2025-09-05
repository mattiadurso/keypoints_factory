import os
import yaml

# wrapper to download files from a URL, empty list means download all
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
            url = file_info["url"]
            dest = file_info["dest"]
            commit = file_info.get("commit", None)

            if url:
                os.system(f"git clone --recursive {url} methods/{dest}")
                # if commit: # or just manually check if commit corresponds
                #     os.system(f"cd methods/{dest} && git checkout {commit}")

            if file_info.get("submodules"):
                for submodule in file_info["submodules"]:
                    os.system(submodule)
