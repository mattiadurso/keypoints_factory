# HPatches full image sequences (~1.5 GB)
# wget -P benchmarks/hpatches https://huggingface.co/datasets/vbalnt/hpatches/resolve/main/hpatches-sequences-release.zip

# # Unzip and remove the 8 high-res sequences to get 108 sequences
# unzip benchmarks/hpatches/hpatches-sequences-release.zip -d benchmarks/hpatches/data
# cd benchmarks/hpatches/data/hpatches-sequences-release
# rm -rf i_contruction i_crownnight i_dc i_pencils i_whitebuilding v_artisans v_astronautis v_talent
# rm -rf benchmarks/hpatches/hpatches-sequences-release.zip

# here the ready-to-use version 
wget -c -O benchmarks/hpatches/hpatches-sequences-release.zip \
  "https://cloud.tugraz.at/index.php/s/gJdwBXtzXCgNc96/download"

mkdir -p benchmarks/hpatches/data
unzip benchmarks/hpatches/hpatches-sequences-release.zip -d benchmarks/hpatches/data
rm benchmarks/hpatches/hpatches-sequences-release.zip