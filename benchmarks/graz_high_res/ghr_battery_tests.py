import os
import glob
from pathlib import Path

wrappers_list = ['aliked', 'disk', 'superpoint', 'ripe', 'dedode'] # 'aliked', 'disk', 'superpoint', 'ripe', 'dedode', 

# common parameters
matcher= 'mnn' 
script_name = 'benchmarks/graz_high_res/ghr_benchmark.py'
min_score = 0.5
ratio_test = 1
stats = True
partial = False
tag = None

# scale factor
scale_factors = {'4K': 1, 'QHD': 1.5, 'FHD': 2, 'HD': 3}
resolutions = ['4K', 'QHD']  # '4K', 'QHD', 'FHD', 'HD'

# --------------------- Kpts budget: 2048 -------------------------------
max_kpts = 2048
th = 0.75

for s in resolutions:
    for wrapper in wrappers_list:
        os.system(f'python {script_name} --wrapper-name {wrapper} --max-kpts {max_kpts} --matcher {matcher} --run-tag {tag} \
            --th {th} --min-score {min_score} --ratio-test {ratio_test} --stats {stats} --partial {partial} --scale-factor {scale_factors[s]}')

        os.system(f'python {script_name} --wrapper-name {wrapper} --max-kpts {max_kpts} --matcher {matcher} --run-tag {tag} \
            --th {th} --min-score {min_score} --ratio-test {ratio_test} --stats {stats} --partial {partial} --scale-factor {scale_factors[s]} \
            --custom-desc sandesc_models/{wrapper}/final.pth')

