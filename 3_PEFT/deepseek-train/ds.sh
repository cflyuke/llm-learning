# sft
# python main.py \
#     --task=sft_train \
#     --bf16 \
#     --checkpoint_dir=outputs/Qwen-0.5B-SFT-FirstHalf \
#     --per_device_train_batch_size=8 \
#     --save_strategy=epoch \
#     --epochs=1 \
#     --split_half=first_half \

# grpo
# python main.py \
#     --task=grpo_train \
#     --model_name_or_path=outputs/Qwen-0.5B-SFT-FirstHalf/checkpoint-117 \
#     --bf16 \
#     --use_vllm \
#     --vllm_mode=colocate \
#     --checkpoint_dir=outputs/Qwen-0.5B-GRPO-SecondHalf \
#     --per_device_train_batch_size=8 \
#     --save_strategy=epoch \
#     --split_half=second_half \

# inference
python main.py \
    --task=inference \
    --checkpoint_dir=outputs/Qwen-0.5B-GRPO-SecondHalf/checkpoint-934