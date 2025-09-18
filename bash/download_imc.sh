# web page for images: https://www.cs.ubc.ca/~kmyi/imw2020/data.html -> test data -> all sequences
# web page for images and gt: https://www.cs.ubc.ca/research/image-matching-challenge/2021/data/
# btw for some reason, the first link downloads all the images (9x100), 
# while the second has gt for all images but is missing: 
# - 1 image  in london_bridge
# - 2 images in florence_cathedral_side
# - 3 images in piazza_san_marco
# I manually fixed this and upload a corrected version downloadable with this script. I also add the validation set.


# test and val set
wget -c -O benchmarks/imc/phototourism.zip \
  "https://cloud.tugraz.at/index.php/s/PTgmwHpkJ5sP3cG/download"
  
mkdir -p benchmarks/imc/data
unzip benchmarks/imc/phototourism.zip -d benchmarks/imc/data
rm benchmarks/imc/phototourism.zip


# then download imc repo
mkdir benchmarks/imc/image-matching-benchmark
git clone https://github.com/ubc-vision/image-matching-benchmark.git benchmarks/imc/image-matching-benchmark
cd benchmarks/imc/image-matching-benchmark && git submodule update --init