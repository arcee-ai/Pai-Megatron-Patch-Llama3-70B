#!/bin/bash

# Environment variables hardcoded
ENV=dsw                                                                         # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=../../                                                      # Path to Megatron Patch code
MODEL_SIZE=70B                                                                  # Model size: 7B, 13B
BATCH_SIZE=1                                                                    # Samples per iteration per GPU: 4, 8
GLOBAL_BATCH_SIZE=128                                                           # Global batch size
LR=1e-5                                                                         # Learning rate: 1e-5, 5e-5
MIN_LR=1e-6                                                                     # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=128                                                                     # Sequence length
PAD_LEN=128                                                                     # Padding length: 100
EXTRA_VOCAB_SIZE=256                                                            # Extra vocabulary size
PR=bf16                                                                         # Precision: fp16, bf16
TP=8                                                                            # Tensor Parallelism
PP=4                                                                            # Pipeline Parallelism
AC=sel                                                                          # Activation checkpoint mode: sel, full
DO=true                                                                         # Use Megatron Zero-1 optimizer: true, false
FL=false                                                                        # Use Flash Attention: true, false
SP=false                                                                        # Use sequence parallelism: true, false
TE=false                                                                        # Use Transformer Engine: true, false
MOE=false                                                                       # Enable MOE: true, false
SAVE_INTERVAL=5000                                                              # Checkpoint save interval
DATASET_PATH=/mnt/llama3-datasets/wudao_llama3bpe_content_document              # Training dataset path
PRETRAIN_CHECKPOINT_PATH=/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1     # Pretrained model path
TRAIN_TOKENS=100000000                    # Number of training tokens
WARMUP_TOKENS=10000                       # Number of warmup tokens
OUTPUT_BASEPATH=/mnt/output_mcore_llama3  # Output path for training results

# Running the pretrain script
bash run_pretrain_mcore_llama.sh \
  $ENV \
  $MEGATRON_PATCH_PATH \
  $MODEL_SIZE \
  $BATCH_SIZE \
  $GLOBAL_BATCH_SIZE \
  $LR \
  $MIN_LR \
  $SEQ_LEN \
  $PAD_LEN \
  $EXTRA_VOCAB_SIZE \
  $PR \
  $TP \
  $PP \
  $AC \
  $DO \
  $FL \
  $SP \
  $TE \
  $MOE \
  $SAVE_INTERVAL \
  $DATASET_PATH \
  $PRETRAIN_CHECKPOINT_PATH \
  $TRAIN_TOKENS \
  $WARMUP_TOKENS \
  $OUTPUT_BASEPATH
