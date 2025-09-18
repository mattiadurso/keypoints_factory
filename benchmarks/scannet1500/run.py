import os

wrappers_list = ["aliked", "disk", "superpoint", "ripe", "dedode", "dedode-G"]

# common parameters
script_name = "benchmarks/benchmark_parallel.py"

min_score = 0.5
ratio_test = 1
stats = True
tag = None
partial = False  # True to run only first scene, quick run for debugging


max_kpts = 2048
th = 0.75

for wrapper in wrappers_list:
    os.system(
        f"python {script_name} --benchmark-name sc \
        --wrapper-name {wrapper} --max-kpts {max_kpts} \
        --run-tag {tag} --ransac-th {th} --min-score {min_score} \
        --ratio-test {ratio_test} --stats {stats}"
    )

    # # SANDesc is not published yet
    # os.system(
    #     f"python {script_name} --benchmark-name sc \
    #     --wrapper-name {wrapper} --max-kpts {max_kpts} \
    #     --run-tag {tag} --ransac-th {th} --min-score {min_score} \
    #     --ratio-test {ratio_test} --stats {stats} \
    #     --custom-desc sandesc_models/{wrapper}/final.pth"
    # )
