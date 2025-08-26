#!/bin/bash

# Set environment variables
export CUDA_VISIBLE_DEVICES=0,1,2,3

# Create output directory
mkdir -p out

# Configuration
MODEL_NAME="Llama-2-13b-hf"
DATASET_NAME="SFT-EKG"
PRED_NAME="PRED-EKG"
TEMPLATE="ekg"
FINETUNING_TYPE="lora"
LORA_TARGET="q_proj,v_proj"
LEARNING_RATE_SFT="5e-5"
LEARNING_RATE_DPO="1e-5"
NUM_EPOCHS_SFT="3.0"
NUM_EPOCHS_DPO="1.0"
BATCH_SIZE_SFT="4"
BATCH_SIZE_DPO="2"
GRADIENT_ACCUMULATION_STEPS="4"
MAX_SAMPLES="20000"

# Paths
BACKBONE_MODEL_PATH="../backbone_model/${MODEL_NAME}"
SFT_OUTPUT_DIR="./out/SFT-${DATASET_NAME}-${MODEL_NAME}"
GENERATED_KNOWLEDGE_DIR="./out/Generated-${DATASET_NAME}-${MODEL_NAME}"
DPO_OUTPUT_DIR="./out/DPO-${DATASET_NAME}-${MODEL_NAME}"

echo "Starting DELLM training pipeline..."

# Stage 1: Supervised Fine-Tuning (SFT)
echo "Stage 1: Training SFT model..."
python bash.py \
    --stage sft \
    --do_train \
    --model_name_or_path ${BACKBONE_MODEL_PATH} \
    --dataset ${DATASET_NAME} \
    --template ${TEMPLATE} \
    --finetuning_type ${FINETUNING_TYPE} \
    --lora_target ${LORA_TARGET} \
    --output_dir ${SFT_OUTPUT_DIR} \
    --overwrite_cache \
    --per_device_train_batch_size ${BATCH_SIZE_SFT} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate ${LEARNING_RATE_SFT} \
    --num_train_epochs ${NUM_EPOCHS_SFT} \
    --plot_loss \
    --bf16

# Stage 2: Generate knowledge using SFT model
echo "Stage 2: Generating knowledge using SFT model..."
python bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ${BACKBONE_MODEL_PATH} \
    --adapter_name_or_path ${SFT_OUTPUT_DIR} \
    --dataset ${DATASET_NAME} \
    --template ${TEMPLATE} \
    --finetuning_type ${FINETUNING_TYPE} \
    --output_dir ${GENERATED_KNOWLEDGE_DIR} \
    --per_device_eval_batch_size 1 \
    --max_samples ${MAX_SAMPLES} \
    --predict_with_generate \
    --bf16 \
    --do_sample False \
    --num_return_sequences 1

# Stage 3: Construct DPO data based on database execution and SQL contribution feedback
echo "Stage 3: Constructing DPO data..."
python construct_dpo.py \
    --sft_file ./data/SFT-EKG.json \
    --output_file ./data/DPO-EKG.json \
    --db_root_path ../databases/bird/train/train_databases \
    --sft_model_path ${SFT_OUTPUT_DIR} \
    --openai_api_key ${OPENAI_API_KEY} \
    --generated_knowledge_file ${GENERATED_KNOWLEDGE_DIR}/predictions.jsonl

# Stage 4: DPO training with constructed preference data
echo "Stage 4: Training DPO model..."
python bash.py \
    --stage dpo \
    --do_train \
    --model_name_or_path ${BACKBONE_MODEL_PATH} \
    --adapter_name_or_path ${SFT_OUTPUT_DIR} \
    --create_new_adapter \
    --dataset DPO-EKG \
    --template ${TEMPLATE} \
    --finetuning_type ${FINETUNING_TYPE} \
    --lora_target ${LORA_TARGET} \
    --output_dir ${DPO_OUTPUT_DIR} \
    --per_device_train_batch_size ${BATCH_SIZE_DPO} \
    --gradient_accumulation_steps ${GRADIENT_ACCUMULATION_STEPS} \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 1000 \
    --learning_rate ${LEARNING_RATE_DPO} \
    --num_train_epochs ${NUM_EPOCHS_DPO} \
    --plot_loss \
    --overwrite_output_dir \
    --bf16

# Stage 5: Generate final knowledge using DPO refined model
echo "Stage 5: Generating final knowledge using DPO refined model..."
python bash.py \
    --stage sft \
    --do_predict \
    --model_name_or_path ${BACKBONE_MODEL_PATH} \
    --adapter_name_or_path ${SFT_OUTPUT_DIR},${DPO_OUTPUT_DIR} \
    --dataset ${PRED_NAME} \
    --template ${TEMPLATE} \
    --finetuning_type ${FINETUNING_TYPE} \
    --output_dir ./out/Final-${DATASET_NAME}-${MODEL_NAME} \
    --per_device_eval_batch_size 1 \
    --max_samples ${MAX_SAMPLES} \
    --predict_with_generate \
    --bf16

echo "DELLM training pipeline completed!"
echo "SFT model saved to: ${SFT_OUTPUT_DIR}"
echo "DPO model saved to: ${DPO_OUTPUT_DIR}"
echo "Final knowledge generated to: ./out/Final-${DATASET_NAME}-${MODEL_NAME}"