# original source by RDD: https://drive.google.com/drive/folders/1QgVaqm4iTUCqbWb7_Fi6mX09EHTId0oA?usp=sharing
# in my version I only converted gt information to my setup

# here the ready-to-use version 
wget -c -O benchmarks/megadepth_air2ground/megadepth_air2ground.zip \
  "https://cloud.tugraz.at/index.php/s/84QXDJRStLaRDKD/download"

mkdir -p benchmarks/megadepth_air2ground
unzip benchmarks/megadepth_air2ground/megadepth_air2ground.zip -d benchmarks/megadepth_air2ground
rm benchmarks/megadepth_air2ground/megadepth_air2ground.zip