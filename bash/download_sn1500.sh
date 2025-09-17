# downloading data
wget -P benchmarks/scannet1500 https://cvg-data.inf.ethz.ch/scannet/scannet1500.zip

# unzip and renaming
unzip benchmarks/scannet1500/scannet1500.zip -d benchmarks/scannet1500/
mv benchmarks/scannet1500/scannet1500/ benchmarks/scannet1500/data
rm benchmarks/scannet1500/scannet1500.zip