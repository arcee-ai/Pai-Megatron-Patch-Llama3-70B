### hf-to-megatron

**hf-to-megatron** is a model checkpoint conversion tool designed to enable users to easily convert Hugging Face checkpoints to Megatron format. This allows users to leverage the distributed training capabilities of Megatron-LM for training large language models (LLMs). The converted models are intended to be used with the PAI-Megatron-Patch codebase. The following models are currently supported:

- BLOOM
- LLaMA/Alpaca
- ChatGLM
- Galactica
- GLM
- GLM130B
- Falcon
- StarCoder

The converted models are stored at: `oss://atp-modelzoo/release/models/pai-megatron-patch/`