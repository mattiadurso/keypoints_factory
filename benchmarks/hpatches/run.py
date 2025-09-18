import os

wrappers_list = ["aliked", "disk", "superpoint", "ripe", "dedode", "dedode-G"]

# common parameters
script_name = "benchmarks/hpatches/hpatches_benchmark.py"

min_score = 0.5
ratio_test = 1
stats = True
tag = None

# --------------------- Kpts budget: 2048 -------------------------------
max_kpts = 2048

for wrapper in wrappers_list:
    os.system(
        f"python {script_name} --wrapper-name {wrapper} --max-kpts {max_kpts} \
        --run-tag {tag} --min-score {min_score} --ratio-test {ratio_test} \
        --stats {stats} --stats {stats} \
        "
    )

    # SANDesc is not published yet
    os.system(
        f"python {script_name} --wrapper-name {wrapper} --max-kpts {max_kpts} \
        --run-tag {tag} --min-score {min_score} --ratio-test {ratio_test} \
        --stats {stats} --stats {stats} \
        --custom-desc sandesc_models/{wrapper}/final.pth \
        "
    )
