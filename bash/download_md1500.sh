# downloading data
wget -P benchmarks/megadepth1500 https://cvg-data.inf.ethz.ch/megadepth/megadepth1500.zip

# unzip and renaming
unzip benchmarks/megadepth1500/megadepth1500.zip -d benchmarks/megadepth1500/
mv benchmarks/megadepth1500/megadepth1500/ benchmarks/megadepth1500/data
rm benchmarks/megadepth1500/megadepth1500.zip