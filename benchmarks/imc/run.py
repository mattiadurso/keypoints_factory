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


# for this benchmark, some methods declared some parameters. If not, use default ones
sandesc = "-sandesc"
params = { # after validation tuning, these are the best params
    "disk": {"th": 0.75, "min_score": 0, "ratio_test": 0.95},
    "disk-sandesc": {"th": 0.75, "min_score": 0, "ratio_test": 0.95},
}

scene_set = "val"  # test, val

if scene_set == "val":
    for th in sorted([0.75, 0.85, 0.95, 1.0]):
        for ratio_test in sorted([0.95, 0.98, 1]):
            for min_score in sorted([0, 0.5]):
                for wrapper_name in wrappers_list:
                    os.system(
                        f"python {script_name} --wrapper-name {wrapper_name} --max-kpts 2048 \
                            --th {th} --min-score {min_score} --ratio-test {ratio_test} \
                            --scene-set {scene_set}"
                    )

                    # SANDesc is not published yet
                    os.system(
                        f"python {script_name} --wrapper-name {wrapper_name} --max-kpts 2048 \
                        --th {th} --min-score {min_score} --ratio-test {ratio_test} \
                        --scene-set {scene_set} --custom-desc sandesc_models/{wrapper_name}/final.pth"
                    )

    os.system("rm -rf benchmarks/imc/benchmark-results")  # heavy files and not needed
    os.system("rm -rf benchmarks/imc/benchmark-visualization")  # I don't use them

elif scene_set == "test":
    for wrapper_name in wrappers_list:
        os.system(
            f"python {script_name} --wrapper-name {wrapper_name} --max-kpts 2048 \
                --th {params[wrapper_name]['th']} --min-score {params[wrapper_name]['min_score']} \
                --ratio-test {params[wrapper_name]['ratio_test']} --scene-set {scene_set} \
                "
        )

        # SANDesc is not published yet
        os.system(
           f"python {script_name} --wrapper-name {wrapper_name} --max-kpts 2048 \
            --th {params[wrapper_name+sandesc]['th']} --min-score {params[wrapper_name+sandesc]['min_score']} \
            --ratio-test {params[wrapper_name+sandesc]['ratio_test']} --scene-set {scene_set} \ 
            --custom-desc sandesc_models/{wrapper_name}/final.pth"
        )

    os.system("rm -rf benchmarks/imc/benchmark-results")  # heavy files and not needed


## Eventually it's possible to delete also the computed matches since it will
## grow a lot, namely ~1GB per sub-folder, and is not needed anymore once the
## benchmark is done unless one wants to run with multiple matching parameters,
## even though is not the spirit of the benchmark. This should be eventually
## done on the validation set which is not provided yet.

# os.system("rm -rf benchmarks/imc/to_import_imc")
