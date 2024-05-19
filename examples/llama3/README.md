# Table of Contents
   * [Installation](#Installation)
   * [Datasets & Model Download](#Datasets-and-Model-Download)
   * [Megatron-LM-Dense Model Training Process](#Megatron-LM-Dense-Model-Training-Process)
      * [Model Format Conversion](#Megatron-LM-Dense-Model-Format-Conversion)
      * [Continue Pretraining](#Megatron-LM-Dense-Continue-Pretraining)
      * [Instruction Fine-Tuning](#Megatron-LM-Dense-Instruction-Fine-Tuning)
   * [Megatron-Core-Dense Model Training Process](#Megatron-Core-Dense-Model-Training-Process)
      * [Model Format Conversion](#Megatron-Core-Dense-Model-Format-Conversion)
      * [Continue Pretraining](#Megatron-Core-Dense-Continue-Pretraining)
      * [Instruction Fine-Tuning](#Megatron-Core-Dense-Instruction-Fine-Tuning)
   * [Downstream Task Evaluation](#Downstream-Task-Evaluation)
      * [Convert Megatron-LM Model Format to HuggingFace](#Convert-Megatron-LM-Model-Format-to-HuggingFace)
      * [Run Evaluation Tools](#Run-Evaluation-Tools)

# Installation
It is recommended to use the official NVIDIA container `nvcr.io/nvidia/pytorch:23.12-py3` to create the container.

```bash
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
cd Pai-Megatron-Patch
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple/
```

# Datasets and Model Download
```bash
cd /mnt
mkdir llama3-ckpts
cd llama3-ckpts
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-ckpts/Meta-Llama-3-8B.tgz
tar -zxf Meta-Llama-3-8B.tgz

mkdir llama3-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.idx

wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-valid.json
```

# Megatron-LM-Dense Model Training Process

Run the `hf2megatron_convertor.sh` script with the following parameters:
```
MEGATRON_PATH=$1                   # Path to Megatron-LM
SOURCE_CKPT_PATH=$2                # Path to the original CKPT
TARGET_CKPT_PATH=$3                # Path to the target CKPT
TP=$4                              # Tensor Parallelism
PP=$5                              # Pipeline Parallelism
MN=$6                              # Model name, e.g., llama3-8b 
EXTRA_VOCAB_SIZE=$7                # Extra vocabulary size
mg2hf=$8                           # Whether to execute mg2hf conversion
```

Run the `run_pretrain_megatron_llama.sh` script with the following parameters:
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Path to Megatron Patch code
MODEL_SIZE=$3                   # Model size: 7B, 13B
BATCH_SIZE=$4                   # Samples per iteration per GPU: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Extra vocabulary size
PR=${11}                        # Precision: fp16, bf16
TP=${12}                        # Tensor Parallelism
PP=${13}                        # Pipeline Parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Use Megatron Zero-1 optimizer: true, false
FL=${16}                        # Use Flash Attention: true, false
SP=${17}                        # Use sequence parallelism: true, false
TE=${18}                        # Use Transformer Engine: true, false
SAVE_INTERVAL=${19}             # Checkpoint save interval
DATASET_PATH=${20}              # Training dataset path
PRETRAIN_CHECKPOINT_PATH=${21}  # Pretrained model path
TRAIN_TOKENS=${22}              # Number of training tokens
WARMUP_TOKENS=${23}             # Number of warmup tokens
OUTPUT_BASEPATH=${24}           # Output path for training results
```

Run the `run_finetune_megatron_llama_withGA.sh` script with the following parameters:
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Path to Megatron Patch code
MODEL_SIZE=$3                   # Model size: 7B, 13B
BATCH_SIZE=$4                   # Samples per iteration per GPU: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Extra vocabulary size
PR=${11}                        # Precision: fp16, bf16
TP=${12}                        # Tensor Parallelism
PP=${13}                        # Pipeline Parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Use Megatron Zero-1 optimizer: true, false
FL=${16}                        # Use Flash Attention: true, false
SP=${17}                        # Use sequence parallelism: true, false
TE=${18}                        # Use Transformer Engine: true, false
SAVE_INTERVAL=${19}             # Checkpoint save interval
DATASET_PATH=${20}              # Training dataset path
VALID_DATASET_PATH=${21}        # Validation dataset path
PRETRAIN_CHECKPOINT_PATH=${22}  # Pretrained model path
TRAIN_ITERS=${23}               # Number of training steps
WARMUP_ITERS=${24}              # Number of warmup steps
OUTPUT_BASEPATH=${25}           # Output path for training results
```

## Megatron-LM-Dense Model Format Conversion
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
sh hf2megatron_convertor.sh \
../../../     \
/mnt/llama3-ckpts/Meta-Llama-3-8B    \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-megatron-tp4-pp1  \
4  \
1  \
llama3-8b \
0 \
false
```

## Megatron-LM-Dense Continue Pretraining
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3
sh run_pretrain_megatron_llama.sh  \
dsw  \
../../ \
8B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
256   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
100000  \
/mnt/llama3-datasets/wudao_llama3bpe_content_document  \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-megatron-tp4-pp1  \
100000000   \
10000   \
/mnt/output_megatron_llama3
```

## Megatron-LM-Dense Instruction Fine-Tuning
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3
sh run_finetune_megatron_llama_withGA.sh  \
dsw  \
../../ \
8B     \
1      \
32     \
1e-5   \
1e-6   \
128   \
128     \
256      \
bf16   \
4      \
1      \
sel    \
true   \
false  \
false  \
false \
100 \
/mnt/llama3-datasets/alpaca_zh-llama2-train.json   \
/mnt/llama3-datasets/alpaca_zh-llama2-valid.json   \
/mnt/llama3-ck

pts/Meta-Llama-3-8B-to-megatron-tp4-pp1  \
1000 \
10 \
/mnt/output_megatron_llama3/
```

# Megatron-Core-Dense Model Training Process

Run the `hf2mcore_convertor.sh` script with the following parameters:
```
MODEL_SIZE=$1                  # Model size: 7B, 13B, 70B
HG_CKPT_PATH=$2                # Path to HF checkpoint
MEGATRON_PATH=$3               # Path to Megatron-LM root directory
SOURCE_CKPT_PATH=$4            # Source path
TARGET_CKPT_PATH=$5            # Target path
TP=$6                          # Tensor Parallelism
PP=$7                          # Pipeline Parallelism
EXTRA_VOCAB_SIZE=$8            # Extra vocabulary size
NUM_EXPERTS=$9                 # Number of experts
EXPERTS_TOPK=${10}             # Topk for expert routing
EP=${11}                       # Expert parallelism
mg2hf=${12}                    # Whether to execute mcore2hf conversion
```

Run the `run_pretrain_mcore_llama.sh` script with the following parameters:
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Path to Megatron Patch code
MODEL_SIZE=$3                   # Model size: 7B, 13B
BATCH_SIZE=$4                   # Samples per iteration per GPU: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Extra vocabulary size
PR=${11}                        # Precision: fp16, bf16
TP=${12}                        # Tensor Parallelism
PP=${13}                        # Pipeline Parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Use Megatron Zero-1 optimizer: true, false
FL=${16}                        # Use Flash Attention: true, false
SP=${17}                        # Use sequence parallelism: true, false
TE=${18}                        # Use Transformer Engine: true, false
MOE=${19}                       # Enable MOE: true, false
SAVE_INTERVAL=${20}             # Checkpoint save interval
DATASET_PATH=${21}              # Training dataset path
PRETRAIN_CHECKPOINT_PATH=${22}  # Pretrained model path
TRAIN_TOKENS=${23}              # Number of training tokens
WARMUP_TOKENS=${24}             # Number of warmup tokens
OUTPUT_BASEPATH=${25}           # Output path for training results
```

Run the `run_finetune_mcore_llama_withGA.sh` script with the following parameters:
```
ENV=$1                          # Running environment: dlc, dsw
MEGATRON_PATCH_PATH=$2          # Path to Megatron Patch code
MODEL_SIZE=$3                   # Model size: 7B, 13B
BATCH_SIZE=$4                   # Samples per iteration per GPU: 4, 8
GLOBAL_BATCH_SIZE=$5            # Global batch size
LR=$6                           # Learning rate: 1e-5, 5e-5
MIN_LR=$7                       # Minimum learning rate: 1e-6, 5e-6
SEQ_LEN=$8                      # Sequence length
PAD_LEN=$9                      # Padding length: 100
EXTRA_VOCAB_SIZE=${10}          # Extra vocabulary size
PR=${11}                        # Precision: fp16, bf16
TP=${12}                        # Tensor Parallelism
PP=${13}                        # Pipeline Parallelism
AC=${14}                        # Activation checkpoint mode: sel, full
DO=${15}                        # Use Megatron Zero-1 optimizer: true, false
FL=${16}                        # Use Flash Attention: true, false
SP=${17}                        # Use sequence parallelism: true, false
TE=${18}                        # Use Transformer Engine: true, false
MOE=${19}                       # Enable MOE: true, false
SAVE_INTERVAL=${20}             # Checkpoint save interval
DATASET_PATH=${21}              # Training dataset path
VALID_DATASET_PATH=${22}        # Validation dataset path
PRETRAIN_CHECKPOINT_PATH=${23}  # Pretrained model path
TRAIN_ITERS=${24}               # Number of training steps
WARMUP_ITERS=${25}              # Number of warmup steps
OUTPUT_BASEPATH=${26}           # Output path for training results
```

## Megatron-Core-Dense Model Format Conversion
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
sh hf2mcore_convertor.sh \
8B \
/mnt/llama3-ckpts/Meta-Llama-3-8B    \
../../../     \
/mnt/llama3-ckpts/Meta-Llama-3-8B    \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1  \
4  \
1  \
256  \
0  \
0  \
0 \
false
```

## Megatron-Core-Dense Continue Pretraining
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3
sh run_pretrain_mcore_llama.sh  \
dsw  \
../../ \
8B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
256   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
false \
100000  \
/mnt/llama3-datasets/wudao_llama3bpe_content_document   \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1  \
100000000   \
10000   \
/mnt/output_mcore_llama3
```

## Megatron-Core-Dense Instruction Fine-Tuning
```bash
cd /workspace/Pai-Megatron-Patch/examples/llama3
sh run_finetune_mcore_llama_withGA.sh  \
dsw  \
../../ \
8B   \
1    \
8 \
1e-5   \
1e-6   \
128  \
128  \
256   \
bf16  \
4   \
1  \
sel  \
true   \
false  \
false   \
false   \
false \
100000  \
/mnt/llama3-datasets/alpaca_zh-llama3-train.json   \
/mnt/llama3-datasets/alpaca_zh-llama3-valid.json   \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1  \
100000000   \
10000   \
/mnt/output_mcore_llama3
```

# Downstream Task Evaluation

## Convert Megatron-LM Model Format to HuggingFace
```bash
cd /workspace/Pai-Megatron-Patch/toolkits/model_checkpoints_convertor/llama
sh hf2megatron_convertor.sh \
../../../     \
/mnt/llama3-ckpts/Meta-Llama-3-8B-to-mcore-tp4-pp1/release  \
/mnt/llama3-ckpts/Meta-Llama-3-8B-hf-megatron-to-hf    \
4  \
1  \
llama3-8b \
0 \
true
```

Please copy the `.json` files (except `pytorch_model.bin.index.json`) from the HuggingFace model folder to the `/mnt/llama3-ckpts/Meta-Llama-3-8B-hf-megatron-to-hf` directory to ensure the model works properly.

## Run Evaluation Tools
```bash
cd /workspace/Pai-Megatron-Patch/LM-Evaluation-Harness-240310
accelerate launch --main_process_port 29051 -m lm_eval \
--model hf \
--model_args pretrained=/mnt/llama3-ckpts/Meta-Llama-3-8B-hf-megatron-to-hf,trust_remote_code=True \
--tasks mmlu,ceval-valid  \
--batch_size 16
```