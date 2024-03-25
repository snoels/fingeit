# FinGeitje

This is a receipt as described in the [alignment handbook](https://github.com/huggingface/alignment-handbook), but for fingeitje

## QLoRa training examples

### SFT
```shell
ACCELERATE_LOG_LEVEL=info accelerate launch --config_file recipes/accelerate_configs/multi_gpu.yaml --num_processes=1 scripts/run_sft.py recipes/fingeit/sft/config_qlora.yaml --load_in_4bit=true --report_to=wandb
```

whit multiple GPUs you should update `--num_process`
