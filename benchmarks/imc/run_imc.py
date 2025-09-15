import os

wrappers_list = ["aliked", "superpoint", "disk", "ripe", "dedode", "dedode-G"]

# Choose the benchmark script to run
script_name = "benchmarks/imc/imc_benchmark.py"

# common parameters
min_score = 0.5
ratio_test = 0.98
stats = True
tag = None
max_kpts = 2048
th = 0.75

for wrapper_name in wrappers_list:
    os.system(
        f"python {script_name} --wrapper-name {wrapper_name} --max-kpts {max_kpts} --run-tag {tag} \
        --th {th} --min-score {min_score} --ratio-test {ratio_test}"
    )

    # SANDesc is not published yet
    os.system(
        f"python {script_name} --wrapper-name dedode --max-kpts {max_kpts} --run-tag {tag} \
        --th {th} --min-score {min_score} --ratio-test {ratio_test} \
        --custom-desc sandesc_models/dedode/final.pth"
    )

os.system("rm -rf benchmarks/imc/benchmark-results")  # heavy files and not needed
os.system("rm -rf benchmarks/imc/benchmark-visualization")  # empty folders


## Eventually it's possible to delete also the computed matches since it will
## grow a lot, namely ~1GB per sub-folder, and is not needed anymore once the
## benchmark is done unless one wants to run with multiple matching parameters,
## even though is not the spirit of the benchmark. This should be eventually
## done on the validation set (no provided yet).

# os.system("rm -rf benchmarks/imc/to_import_imc")
