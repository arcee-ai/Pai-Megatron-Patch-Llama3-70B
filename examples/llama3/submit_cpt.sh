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
PP=1                                                                            # Pipeline Parallelism
AC=sel                                                                          # Activation checkpoint mode: sel, full
DO=true                                                                         # Use Megatron Zero-1 optimizer: true, false
FL=false                                                                        # Use Flash Attention: true, false
SP=false                                                                        # Use sequence parallelism: true, false
TE=true                                                                         # Use Transformer Engine: true, false
MOE=false                                                                       # Enable MOE: true, false
SAVE_INTERVAL=5000                                                              # Checkpoint save interval
DATASET_PATH=/workspace/llama3-70B-checkpoints/llama3_data/SlimPajama_llamabpe_text_document     # Training dataset path
PRETRAIN_CHECKPOINT_PATH=/workspace/llama3-70B-checkpoints/Meta-Llama-3-70B-to-mcore-tp8-pp4     # Pretrained model path
TRAIN_TOKENS=100000000                                                                           # Number of training tokens
WARMUP_TOKENS=10000                                                                              # Number of warmup tokens
OUTPUT_BASEPATH=/workspace/llama3-70B-checkpoints/trained_ckpt  # Output path for training results

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
