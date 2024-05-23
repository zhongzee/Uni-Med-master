import subprocess
import os

def run_training_script():
    # Set CUDA_VISIBLE_DEVICES environment variable
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'

    # Build the command to execute
    command = [
        "/root/miniconda3/envs/Tella/bin/torchrun",
        "--nproc_per_node=1",
        "--master_port", "18340",
        "train.py",  # This should be the name of your training script
        "--Train_csv_path", "/root/autodl-tmp/Uni/train.csv",
        "--Eval_csv_path", "/root/autodl-tmp/Uni/test.csv",
        "--output_dir", "/root/autodl-tmp/Uni/Results/PMC_llama7B_PMCCLIP_QA_pretrain_no_aug-test1",
        "--run_name", "PMC_llama7B_PMCCLIP_QA_pretrain_no_aug-test1",
        "--num_train_epochs", "30",
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
        "--deepspeed", "./ds_config/ds_config_zero3_offload.json",
        "--checkpointing", "false",
        "--bf16", "True",
        "--tf32", "True"
    ]

    # Execute the command
    subprocess.run(command)

if __name__ == "__main__":
    run_training_script()
