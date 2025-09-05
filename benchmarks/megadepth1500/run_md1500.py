import os


wrappers_list = ['aliked', 'disk', 'superpoint', 'ripe', 'dedode', 'dedode-G']

# common parameters
script_name = 'benchmarks/megadepth1500/md1500_benchmark.py'
min_score = 0.5
ratio_test = 1
stats = True
tag = None

# --------------------- Kpts budget: 2048 -------------------------------
max_kpts = 2048
th = 0.75

for wrapper in wrappers_list:
    os.system(f'python {script_name} --wrapper-name {wrapper} --max-kpts {max_kpts} --run-tag {tag} \
        --th {th} --min-score {min_score} --ratio-test {ratio_test} --stats {stats}')

    # # SANDesc is not published yet
    # os.system(f'python {script_name} --wrapper-name {wrapper} --max-kpts {max_kpts} --run-tag {tag} \
    #     --th {th} --min-score {min_score} --ratio-test {ratio_test} --stats {stats} \
    #     --custom-desc sandesc_models/{wrapper}/final.pth')


# --------------------- Kpts budget: 30 000 ------------------------------
max_kpts = 30_000
th = 0.25

for wrapper in wrappers_list:
    os.system(f'python {script_name} --wrapper-name {wrapper} --max-kpts {max_kpts} --run-tag {tag} \
        --th {th} --min-score {min_score} --ratio-test {ratio_test} --stats {stats}')

    # # SANDesc is not published yet
    # os.system(f'python {script_name} --wrapper-name {wrapper} --max-kpts {max_kpts} --run-tag {tag} \
    #     --th {th} --min-score {min_score} --ratio-test {ratio_test} --stats {stats} \
    #     --custom-desc sandesc_models/{wrapper}/final.pth')
