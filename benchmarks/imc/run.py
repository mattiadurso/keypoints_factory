import os


wrappers_list = [
    "aliked",
    "disk",
    "superpoint",
    "ripe",
    "dedode",
    "dedode-G",
]

# Choose the benchmark script to run
script_name = "benchmarks/imc/imc_benchmark.py"

scene_set = "val"  # test, val
min_score = 0.0
ratio_test = 1.0
th = 1.0
max_kpts = 2048
multiview = False

for wrapper_name in wrappers_list:
    os.system(
        f"python {script_name} --wrapper-name {wrapper_name} --max-kpts 2048 \
            --th {th} --min-score {min_score} --ratio-test {ratio_test} \
            --scene-set {scene_set} --multiview {multiview}"
    )

    # SANDesc is not published yet
    os.system(
        f"python {script_name} --wrapper-name {wrapper_name} --max-kpts 2048 \
        --th {th} --min-score {min_score} --ratio-test {ratio_test} --multiview {multiview} \
        --scene-set {scene_set} --custom-desc sandesc_models/{wrapper_name}/final.pth"
    )

os.system("rm -rf benchmarks/imc/benchmark-results")  # heavy files and not needed
os.system("rm -rf benchmarks/imc/benchmark-visualization")  # I don't use them


## Eventually it's possible to delete also the computed matches since it will
## grow a lot, namely ~1GB per sub-folder, and is not needed anymore once the
## benchmark is done unless one wants to run with multiple matching parameters,
## even though is not the spirit of the benchmark. This should be eventually
## done on the validation set which is not provided yet.

# os.system("rm -rf benchmarks/imc/to_import_imc")
