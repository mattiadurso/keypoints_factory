wget -c -O benchmarks/graz_high_res/GrazHighResolution.zip \
  "https://cloud.tugraz.at/index.php/s/pWq9q62c9g5npCG/download"

mkdir -p benchmarks/graz_high_res/data
unzip benchmarks/graz_high_res/GrazHighResolution.zip -d benchmarks/graz_high_res
rm benchmarks/graz_high_res/GrazHighResolution.zip
  