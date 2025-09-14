import os

wrappers_list = ["aliked", "disk", "superpoint", "ripe", "dedode", "dedode-G"]

# common parameters
# Choose the benchmark script to run
# script_name = 'benchmarks/graz_high_res/ghr_benchmark.py'
# NOTICE: results might be slightly different due to parallelism.
# Due to the latter, and approach 'first extract, then matche them
# all' it's much faster.
script_name = "benchmarks/graz_high_res/ghr_benchmark_parallel.py"

min_score = 0.5
ratio_test = 1
stats = True
partial = False  # True to run only first scene, quick run for debugging
tag = None

# scale factor
scale_factors = {"4K": 1, "QHD": 1.5, "FHD": 2}
resolutions = ["4K", "QHD", "FHD"]  # '4K', 'QHD', 'FHD'

# --------------------- Kpts budget: 2048 -------------------------------
max_kpts = 2048
th = 0.75

for s in resolutions:
    for wrapper in wrappers_list:
        os.system(
            f"python {script_name} --wrapper-name {wrapper} --max-kpts {max_kpts} \
                  --run-tag {tag} --th {th} --min-score {min_score} --ratio-test {ratio_test} \
                  --stats {stats} --partial {partial} --scale-factor {scale_factors[s]}"
        )

        # # SANDesc is not published yet
        # os.system(f'python {script_name} --wrapper-name {wrapper} --max-kpts {max_kpts} \
        #           --run-tag {tag} --th {th} --min-score {min_score} --ratio-test {ratio_test} \
        #           --stats {stats} --partial {partial} --scale-factor {scale_factors[s]} \
        #           --custom-desc sandesc_models/{wrapper}/final.pth')
