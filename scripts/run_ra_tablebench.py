import os
import subprocess


base_dir = "/your_fs/your_username/RankAgent-Benchmark/"
all_data_names = [
    "tablebench",
]

for cur_data_name in all_data_names:
    num_gpus = 0

    config_path = os.path.join(
        base_dir,
        "configs",
        "3rd_party_benchmarks",
        f"{cur_data_name}",
    )
    # config_name = f"ra_tablebench_deepseek_v3_0324"
    config_name = f"ra_tablebench_llama405b"
    print(f"\nRunning with config: {config_path}")
    job_name = f"{cur_data_name}_{config_name}"
    script_path = f"./scripts/{cur_data_name}/{job_name}.slurm"
    # get the folder path and create it if it doesn't exist
    folder_path = os.path.dirname(script_path)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    with open(script_path, "w") as f:
        lines_to_write = [
            "#!/bin/bash\n",
            "#\n",
            # f"#SBATCH --qos={qos}\n",
            f"#SBATCH --chdir={base_dir}\n",
            f"#SBATCH --gres=gpu:{num_gpus}\n",
            "#SBATCH --mem 128G\n",
            "#SBATCH -c 64\n",
            f"#SBATCH --job-name={job_name}\n",
            "#SBATCH --mem 128G\n",
            f"#SBATCH --output=/your_fs/your_username/RankAgent-Benchmark/slurm/{cur_data_name}/{job_name}.stdout\n",
            f"#SBATCH --error=/your_fs/your_username/RankAgent-Benchmark/slurm/{cur_data_name}/{job_name}.stderr\n",
            "\n",
            f"python rankagent_tablebench.py --config-path={config_path} --config-name={config_name}\n",
        ]
        for cur_line in lines_to_write:
            f.write(cur_line)
        f.close()

    subprocess.run(
        [
            "sbatch",
            f"{script_path}",
        ]
    )
    print(f"Submitted task for {job_name}\n")
