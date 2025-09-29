# original source by RDD: https://drive.google.com/drive/folders/1QgVaqm4iTUCqbWb7_Fi6mX09EHTId0oA?usp=sharing
# in my version I only converted gt information to my setup

# here the ready-to-use version 
wget -c -O benchmarks/megadepth_view/megadepth_view.zip \
  "https://cloud.tugraz.at/index.php/s/NQKt8WMzXW52CZk/download"

mkdir -p benchmarks/megadepth_view
unzip benchmarks/megadepth_view/megadepth_view.zip -d benchmarks/megadepth_view
rm benchmarks/megadepth_view/megadepth_view.zip