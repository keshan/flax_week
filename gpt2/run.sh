export MODEL_DIR=Sinhala-gpt2

./run_clm_flax.py \
                --output_dir="${MODEL_DIR}" \
                --model_type="gpt2" \
                --config_name="${MODEL_DIR}" \
                --tokenizer_name="${MODEL_DIR}" \
                --dataset_name="keshan/clean-si-mc4" \
                --dataset_config_name="si" \
                --do_train \
                --do_eval \
                --block_size="512" \
                --per_device_train_batch_size="64" \
                --learning_rate="5e-3" \
                --warmup_steps="1000" \
                --adam_beta1="0.9" \
                --adam_beta2="0.98" \
                --weight_decay="0.01" \
                --overwrite_output_dir \
                --num_train_epochs="20" \
                --logging_steps="500" \
                --save_steps="2500" \
                --eval_steps="2500" \
                --preprocessing_num_workers="80" \
                --push_to_hub
