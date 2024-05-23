# run_train_downstream.py

import sys
import multiprocessing
import train_downstream

def run_training():
    train_downstream.main()

if __name__ == "__main__":
    # multiprocessing.set_start_method('spawn', force=True)
    sys.argv = [
        "train_downstream.py",
        "--Train_csv_path", "/root/autodl-tmp/VQA_RAD/Task_COT_New_related_final.csv",
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
        # 其他参数...
    ]

    p = multiprocessing.Process(target=run_training)
    p.start()
    p.join()