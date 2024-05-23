import subprocess
import os
# which torchrun
# export PATH="/path/to:$PATH"
def run_training_script():
    # 设置 CUDA_VISIBLE_DEVICES 环境变量
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # 构建要执行的命令
    command = [
        "/root/miniconda3/envs/Tella/bin/torchrun",
        "--nproc_per_node=1",
        "--master_port", "18341",
        "train_downstream.py",
        "--Train_csv_path", "/root/autodl-tmp/VQA_RAD/trainset.csv",
        "--Eval_csv_path", "/root/autodl-tmp/VQA_RAD/testset.csv",
        "--output_dir", "./Results/QA_no_pretrain_no_aug_test1",
        "--run_name", "QA_no_pretrain_no_aug",
        "--num_train_epochs", "10",#
        "--per_device_train_batch_size", "1",
        "--per_device_eval_batch_size", "8",
        "--gradient_accumulation_steps", "1",
        "--evaluation_strategy", "epoch",
        "--save_strategy", "epoch",
        "--load_best_model_at_end", "True",
        "--save_total_limit", "2",
        "--learning_rate", "2e-5",
        "--weight_decay", "0.",
        "--warmup_ratio", "0.03",
        "--lr_scheduler_type", "cosine",
        "--logging_steps", "1",
        "--deepspeed", "./ds_config/ds_config_zero2.json",
        "--checkpointing", "False",
        "--bf16", "True",
        "--tf32", "True"
    ]

    # 执行命令
    subprocess.run(command)

if __name__ == "__main__":
    run_training_script()
