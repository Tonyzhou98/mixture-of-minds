import os
import subprocess


base_dir = "/your_fs/your_username/RankAgent-Benchmark/"
qos = "genai_interns_low"
all_data_names = [
    "tablebench",
]
all_model_names = [
    # llama models
    # "llama3p2_1b_instruct",
    # "llama3p2_3b_instruct",
    # "llama3_8b_instruct",
    # "llama3p3_70b_instruct",
    # "llama3p1_405b_instruct",
    # "llama3p1_405b_instruct_fp8",
    # "llama4_scout_instruct",
    # "llama4_maverick_instruct",
    "llama4_maverick_instruct_fp8",
    # deepseek models
    # "llama_70b_deepseek_r1_distill",
    # "deepseek_v2_chat_0628",
    # "deepseek_v3",
    # "deepseek_v3_0324",
    # "deepseek_r1",
    # qwen models
    # "qwen2p5_0p5b_instruct",
    # "qwen2p5_1p5b_instruct",
    # "qwen2p5_3b_instruct",
    # "qwen2p5_7b_instruct",
    # "qwen2p5_14b_instruct",
    # "qwen2p5_32b_instruct",
    # "qwen2p5_72b_instruct",
]

for cur_data_name in all_data_names:
    for cur_model_name in all_model_names:
        if (
            "deepseek" in cur_model_name
            or cur_model_name == "llama3p1_405b_instruct"
            # or cur_model_name == "llama4_scout_instruct"
            # or cur_model_name == "llama4_maverick_instruct",
        ):
            num_gpus = 0  # we serve deepseek models separately
        else:
            num_gpus = 8

        config_path = os.path.join(
            base_dir,
            "configs",
            "3rd_party_benchmarks",
            f"{cur_data_name}",
        )
        config_name = f"{cur_model_name}"
        print(f"\nRunning with config: {config_path}")
        job_name = f"{cur_data_name}_{cur_model_name}"
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
                f"python tablebench_eval.py --config-path={config_path} --config-name={config_name}\n",
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
