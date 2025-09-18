import os


wrappers_list = ["aliked", "disk", "superpoint", "ripe", "dedode", "dedode-G"]

# Choose the benchmark script to run
script_name = "benchmarks/benchmark_parallel.py"

# common parameters
min_score = 0.5
ratio_test = 1
stats = True
tag = None

max_kpts = [2048, 30_000]
ths = [0.75, 0.25]

for wrapper in wrappers_list:
    for max_kpts, th in zip(max_kpts, ths):
        os.system(
            f"python {script_name} --benchmark-name md \
            --wrapper-name {wrapper} --max-kpts {max_kpts} \
            --run-tag {tag} --ransac-th {th} --min-score {min_score} \
            --ratio-test {ratio_test} --stats {stats}"
        )

        # # SANDesc is not published yet
        # os.system(
        #     f"python {script_name} --benchmark-name md \
        #     --wrapper-name {wrapper} --max-kpts {max_kpts} \
        #     --run-tag {tag} --ransac-th {th} --min-score {min_score} \
        #     --ratio-test {ratio_test} --stats {stats} \
        #     --custom-desc sandesc_models/{wrapper}/final.pth"
        # )
