import os


wrappers_list = ["dedode"]

# Choose the benchmark script to run
# script_name = 'benchmarks/megadepth1500/md1500_benchmark.py'
# NOTICE: results might be slightly different due to parallelism.
# Due to the latter, and approach 'first extract, then matche them
# all' it's much faster.
script_name = "benchmarks/megadepth1500/md1500_benchmark_parallel.py"

paths_to_sandesc_models = [
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-47104.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-48128.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-49152.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-50176.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-51200.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-52224.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-53248.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-54272.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-55296.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-56320.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-57344.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-58368.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-59392.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-60416.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-61440.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-62464.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-63488.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-64512.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-65536.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-66560.pth",
    "/home/mattia/Desktop/Repos/sandesc/wandb/latest-run/files/saved_model/checkpoint-iteration-67584.pth",
]

# common parameters
min_score = 0.5
ratio_test = 1
stats = True
tag = None

## --------------------- Kpts budget: 2048 -------------------------------
max_kpts = 2048
th = 0.75


# os.system(
#     f"python {script_name} --wrapper-name dedode-G --max-kpts {max_kpts} --run-tag {tag} \
#     --th {th} --min-score {min_score} --ratio-test {ratio_test} --stats {stats}"
# )

# SANDesc is not published yet
# os.system(
#     f"python {script_name} --wrapper-name dedode --max-kpts {max_kpts} --run-tag {tag} \
#     --th {th} --min-score {min_score} --ratio-test {ratio_test} --stats {stats} \
#     --custom-desc sandesc_models/dedode/final.pth"
# )

for path in paths_to_sandesc_models:
    # SANDesc is not published yet
    os.system(
        f"python {script_name} --wrapper-name dedode --max-kpts {max_kpts} --run-tag {path.split('/')[-1].split('.')[0].split('-')[-1]} \
        --th {th} --min-score {min_score} --ratio-test {ratio_test} --stats {stats} \
        --custom-desc {path} \
        --keypoints-path /home/mattia/Desktop/Repos/wrapper_factory/benchmarks/megadepth1500/intermediate/dedode-G_kpts_2048_20250913_101100/keypoints.pt"
    )


## --------------------- Kpts budget: 30 000 ------------------------------
# max_kpts = 30_000
# th = 0.25

# for wrapper in wrappers_list:
#     os.system(f'python {script_name} --wrapper-name {wrapper} --max-kpts {max_kpts} --run-tag {tag} \
#         --th {th} --min-score {min_score} --ratio-test {ratio_test} --stats {stats}')

#     # SANDesc is not published yet
#     os.system(f'python {script_name} --wrapper-name {wrapper} --max-kpts {max_kpts} --run-tag {tag} \
#         --th {th} --min-score {min_score} --ratio-test {ratio_test} --stats {stats} \
#         --custom-desc sandesc_models/{wrapper}/final.pth')
