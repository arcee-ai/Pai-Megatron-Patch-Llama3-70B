## Data Preprocessing

It is recommended to prepare pretraining data in a DSW instance within the PAI Lingjun Intelligent Computing Service. Here, the preparation process for the Chinese WuDao 2.0 dataset is used as an example to provide data preprocessing guidelines:

Download the WuDaoCorpora2.0 open-source dataset to the `/mnt/workspace/llama3-datasets` working directory. We provide some sample data as an example, which can be downloaded and decompressed with the following commands:

```shell
wget https://atp-modelzoo.oss-cn-hangzhou.aliyuncs.com/release/datasets/WuDaoCorpus2.0_base_sample.tgz
tar zxvf WuDaoCorpus2.0_base_sample.tgz 
```

Assume the decompressed folder is named `wudao_200g`. The format and size of the **original** WuDao dataset in this folder are shown in the screenshot below:
![image.png](https://intranetproxy.alipay.com/skylark/lark/0/2023/png/226643/1681404415062-92c59f2f-380a-4357-baf8-6f626ad5a217.png#clientId=u90be3297-6831-4&from=paste&height=213&id=NOUVT&originHeight=426&originWidth=1054&originalType=binary&ratio=2&rotation=0&showTitle=false&size=154924&status=done&style=none&taskId=ua99b6661-8759-4a1e-8b97-2cd86736261&title=&width=527)

We have prepared a data preprocessing process for Megatron-LM training, which you can choose according to your needs.

### Megatron-LM Training Data Preparation

MMap data is a pre-tokenized data format that can significantly reduce the time spent waiting for data loading during training and fine-tuning, especially when the dataset is very large.

1. Clean the WuDao dataset and convert the file format. You can refer to the following bash script for the specific process, which will generate a consolidated **merged_wudao_cleaned.json**.

```bash
#! /bin/bash
set -ex
# Set the path of the original data here
data_dir=/mnt/workspace/llama3-datasets/wudao_200g

# Start data cleaning process
dataset_dir=$(dirname $data_dir)
mkdir -p ${dataset_dir}/cleaned_wudao_dataset
cd ${dataset_dir}/cleaned_wudao_dataset
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama2-codes/preprocess_wudao2.py
# This differs from the previous section by adding the key parameter set to text
python preprocess_wudao2.py -i ${data_dir} -o ${dataset_dir}/cleaned_wudao_dataset -k text -p 32

# Merge cleaned data
mkdir ${dataset_dir}/wudao
cd ${dataset_dir}/wudao
find ${dataset_dir}/cleaned_wudao_dataset -name "*.json" -exec cat {} + > ${dataset_dir}/wudao/merged_wudao_cleaned.json
rm -rf ${dataset_dir}/cleaned_wudao_dataset
```

After running the script, the internal file structure of `llama3-datasets` will look like this, with a new `wudao` folder:

```bash
llama3-datasets
├── wudao_200g 
└── wudao
    └── merged_wudao_cleaned.json
```

2. Use the **merged_wudao_cleaned.json** file generated in the first step to split the data into several groups and compress them for subsequent multi-threaded processing:

```bash
apt-get update
apt-get install zstd

# Set the number of chunks to 10. If data processing is slow, set it slightly larger
NUM_PIECE=10

# Process the merged_wudao_cleaned.json file
mkdir -p ${dataset_dir}/cleaned_zst/
# Query total length of the data and split it
NUM=$(sed -n '$=' ${dataset_dir}/wudao/merged_wudao_cleaned.json)
echo "Total line of dataset is $NUM, data will be split into $NUM_PIECE pieces for processing"
NUM=`expr $NUM / $NUM_PIECE`
echo "Each group is processing $NUM samples"
split_dir=${dataset_dir}/split
mkdir $split_dir
split -l $NUM --numeric-suffixes --additional-suffix=.jsonl ${dataset_dir}/wudao/merged_wudao_cleaned.json $split_dir/

# Data compression
o_path=${dataset_dir}/cleaned_zst/
mkdir -p $o_path
files=$(ls $split_dir/*.jsonl)
for filename in $files
do
   f=$(basename $filename)
   zstd -z $filename -o $o_path/$f.zst &
done
rm -rf $split_dir
rm ${dataset_dir}/wudao/merged_wudao_cleaned.json
```

After running the script, the internal file structure of `llama3-datasets` will look like this, with a new `cleaned_zst` folder containing 10 compressed files in each subfolder:

```bash
llama3-datasets
├── wudao_200g
├── wudao
└── cleaned_zst
    ├── 00.jsonl.zst
    │   ...
    └── 09.jsonl.zst
```

3. Create MMAP format pretraining dataset.

Go to the [Pai-Megatron-Patch](https://github.com/alibaba/Pai-Megatron-Patch) open-source website to get the Megatron model training tool Pai-Megatron-Patch source code and copy it to the working directory `/mnt/workspace/`.

```bash
# Get the training code from the open-source website
git clone --recurse-submodules https://github.com/alibaba/Pai-Megatron-Patch.git
```

In the DSW Terminal, go to the code directory: `/mnt/workspace/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing`. Check the `run_make_pretraining_dataset.sh` script. It has five startup parameters that need to be input during execution. The parameter list is as follows:

```
MEGATRON_PATCH_PATH=$1             # Set the path of the Megatron Patch code
input_data_dir=$2                  # Path of the packaged wudao dataset folder
tokenizer=$3                       # llamabpe
output_data_dir=$4                 # Directory for output bin and idx files  
load_dir=$5                        # Path of the tokenizer_config.json file
```

Here is an example of running the script:

```bash
# Set the dataset path and working path here
export dataset_dir=/mnt/workspace/llama3-datasets
export WORK_DIR=/mnt/workspace

# Generate mmap format pretraining datasets for training and validation sets
cd ${WORK_DIR}/Pai-Megatron-Patch/toolkits/pretrain_data_preprocessing
bash run_make_pretraining_dataset.sh \
../.. \
${dataset_dir}/cleaned_zst/ \
llamabpe \
${dataset_dir}/ \
${WORK_DIR}/llama3-ckpts/Meta-Llama-3-8B
rm -rf ${dataset_dir}/cleaned_zst
```

After running the script, the internal file structure of `llama3-datasets` will look like this, with two files in the `wudao` folder having the same name but different extensions:

```bash
llama3-datasets
├── wudao_200g
└── wudao
   ├── wudao_llama3bpe_content_document.bin
   └── wudao_llama3bpe_content_document.idx
```

### Small-scale Preprocessed Data Download for Trial Use

To facilitate user trials, we also provide preprocessed small-scale data that can be downloaded and used directly:

```bash
cd /mnt/workspace/llama3-datasets
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.bin
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/wudao_llama3bpe_content_document.idx
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-train.json
wget https://atp-modelzoo-wlcb-pai.oss-cn-wulanchabu.aliyuncs.com/release/models/pai-megatron-patch/llama3-datasets/alpaca_zh-llama3-valid.json
```