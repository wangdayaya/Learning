

# 下载并安装 anaconda

# 创建python 3.10以上的虚拟环境

关键库版本号如下：

- transformers==4.46.0
- optree==0.13.0
- peft==0.7.1
- accelerate==1.0.1
- bitsandbytes==0.44.1
- torch==2.1.2+cu121
- torchaudio==2.1.2+cu121
- torchvision==0.16.2+cu121
- tensorboard==2.18.0
- numpy==1.26.4
- pandas==2.0.3
- datasets==2.18.0
- tokenizers==0.20.1
- xformers==0.0.28.post2
- dataclasses-json==0.6.4



# 安装 cuda
我安装的是 12.1 ，如果机器上面已经有多个 cuda ，记得将环境变量改成使用 12.1 版本。下面是我找的一些可以参考的安装教程。

- 参考:https://blog.csdn.net/changyana/article/details/135876568
- cudn 安装参考：https://developer.nvidia.com/cuda-toolkit-archive
- cudnn 安装参考：https://docs.nvidia.com/deeplearning/cudnn/latest/reference/support-matrix.html
 
 安装成功之后，在 powershell 中打印如下命令查看是否安装成功你想要的指定版本。
 
     nvcc -V


# 安装pytorch

进入官网 https://pytorch.org/get-started/previous-versions/ 找适合自己 cuda 的版本。

为了后面适配 flash 的 windows 版本，这里如下安装：
```
pip install torch==2.1.2 torchvision==0.16.2 torchaudio==2.1.2 --index-url https://download.pytorch.org/whl/cu121
```

安装之后，使用 python 打印出 torch 版本看是否是你想要的。

# 安装flash-attn（可选）

flash-attn 库主要是加速训练的，不想要可以不安装。

在自己虚拟环境中输出：

    pip debug --verbose
    
能看到如下信息：


 
![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/5f0104f70e91436bbcd6820217bc4388~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1731376452&x-orig-sign=s%2FzQIiQL6xFDUxxKvSKND2NtJx0%3D)
    
进入 https://github.com/bdashore3/flash-attention/releases?page=1 ，根据上面 Compatible tags 和安装的 cuda 版本、 torch 版本来选择兼容的 wheel 文件，下载好之后进行安装。我这里根据上面的 cuda 和 pytorch、python 版本选择了：

    [flash_attn-2.4.1+cu121torch2.1cxx11abiFALSE-cp310-cp310-win_amd64.whl](https://github.com/bdashore3/flash-attention/releases/download/v2.4.1/flash_attn-2.4.1+cu121torch2.1cxx11abiFALSE-cp310-cp310-win_amd64.whl)

到下载好该文件的目录下,使用自己的 pip 执行命令：

    pip install "flash_attn-2.4.1+cu121torch2.1cxx11abiFALSE-cp310-cp310-win_amd64.whl"


# 准备好 llama3-8b 原始模型



去官网下载即可。

    https://www.modelscope.cn/models/LLM-Research/Meta-Llama-3-8B


# 准备指令微调的中文数据

原则上要先对 llama3-8b 使用 100G 以上的中文数据进行全参数微调先适应中文，然后再使用中文指令数据进行精调效果才好，但是这里是为了跑通流程，不纠结细节，而且重新全参数微调我也没有设备。下面是一些常用的数据，自己提前下载好。

- [alpaca_zh_51k](https://huggingface.co/datasets/hfl/alpaca_zh_51k)
- [stem_zh_instruction](https://huggingface.co/datasets/hfl/stem_zh_instruction)
- [ruozhiba_gpt4](https://huggingface.co/datasets/hfl/ruozhiba_gpt4)


# 准好指令微调的训练代码

将下面的仓库拉到本地，我们只需要使用用到 scripts/training/run_clm_sft_with_peft.py 文件即可。

    https://github.com/ymcui/Chinese-LLaMA-Alpaca-3


## 参数

这里是我的参数，你只需要把模型和数据位置改成你自己的即可，然后启动上面的脚本进行训练。训练过程比较慢，我这里是 4090 24G 的显卡，比这个低配的可能跑不起来。

```
--model_name_or_path
D:\Llama-3-8B
--tokenizer_name_or_path
D:\Llama-3-8B
--dataset_dir
D:\Chinese-LLaMA-Alpaca-3\data\train
--per_device_train_batch_size
1
--per_device_eval_batch_size
1
--do_train
1
--do_eval
1
--seed
42
--num_train_epochs
1
--lr_scheduler_type
cosine
--learning_rate
1e-4
--warmup_ratio
0.03
--weight_decay
0.1
--save_strategy
steps
--save_total_limit
3
--evaluation_strategy
steps
--eval_steps
100
--save_steps
200
--gradient_accumulation_steps
8
--preprocessing_num_workers
8
--max_seq_length
512
--output_dir
D:\llama-3-chinese-8b-instruct-lora
--overwrite_output_dir  # 如果想接着之前的 checkpoints 继续训练，设置为 0
1
--lora_rank
8
--lora_alpha
32
--trainable
"q_proj,v_proj,k_proj,o_proj,gate_proj,down_proj,up_proj"
--lora_dropout
0.05
--modules_to_save
"embed_tokens,lm_head"
--bf16
1
--torch_dtype
bfloat16
--validation_file
"D:\Chinese-LLaMA-Alpaca-3\data\eval\eval_2737.json"
--load_in_kbits
16
```


# lora 合并


当你训练完模型之后，得到的只是一个 lora 模型，想要使用还要和原来的 llama 模型进行合并，使用 scripts/merge_llama3_with_chinese_lora_low_mem.py 脚本即可，下面是我使用的参数，你改成自己的即可。这里消耗的是 CPU ，不是 GPU。参数如下：

    --base_model
    D:\Llama-3-8B
    --lora_model
    D:\Chinese-LLaMA-Alpaca-3\lora\checkpoint-1800
    --output_dir
    D:\Chinese-LLaMA-Alpaca-3\lora-merge



会打印如下信息，表示合并成功：

    Base model: D:\Llama-3-8B
    LoRA model: D:\Chinese-LLaMA-Alpaca-3\lora\checkpoint-1800
    Loading D:\Chinese-LLaMA-Alpaca-3\lora\checkpoint-1800
    Loading ckpt model-00001-of-00004.safetensors
    Merging...
    Saving ckpt model-00001-of-00004.safetensors to D:\Chinese-LLaMA-Alpaca-3\lora-merge in HF format...
    Loading ckpt model-00002-of-00004.safetensors
    Merging...
    Saving ckpt model-00002-of-00004.safetensors to D:\Chinese-LLaMA-Alpaca-3\lora-merge in HF format...
    Loading ckpt model-00003-of-00004.safetensors
    Merging...
    Saving ckpt model-00003-of-00004.safetensors to D:\Chinese-LLaMA-Alpaca-3\lora-merge in HF format...
    Loading ckpt model-00004-of-00004.safetensors
    Merging...
    Saving ckpt model-00004-of-00004.safetensors to D:\Chinese-LLaMA-Alpaca-3\lora-merge in HF format...
    Saving tokenizer
    Saving config.json from D:\Llama-3-8B
    Saving generation_config.json from D:\Llama-3-8B
    Saving model.safetensors.index.json from D:\Llama-3-8B
    Done.
    Check output dir: D:\Chinese-LLaMA-Alpaca-3\lora-merge
    
# 量化

这一步主要是为了将合并后的模型进行量化，加速推理。

## windows 需要提前安装 cmake 才能编译 llama.cpp 仓库

https://cmake.org/download/ 下载 [cmake-3.31.0-rc2-windows-x86_64.msi](https://github.com/Kitware/CMake/releases/download/v3.31.0-rc2/cmake-3.31.0-rc2-windows-x86_64.msi)
傻瓜安装，重新打开 powershell 执行 cmake --version 会打印出来版本号，如果没有就将安装位置 D:\CMake\bin 加入系统变量 PATH 。

## 安装 llama.cpp


把仓库拉下来

    git clone https://github.com/ggerganov/llama.cpp



进入目录，然后切换到虚拟环境，执行

     pip install -r .\requirements\requirements-convert_hf_to_gguf.txt
 执行之后torch版本可能会降低到2.2.2 ，如果不想改变 torch 版本可以将 requirements-convert_hf_to_gguf.txt 中的相关内容删除。确保安装好 cmake 然后执行
 
     cmake -B build
     cmake --build build --config Release
这里会生成一系列的命令在 llama.cpp\build\bin\Release 下面。可能会打印出很多 warining ，不必管他。

## 转换 hf 为 gguf

使用 llama.cpp\convert_hf_to_gguf.py 执行以下命令

 
    python convert_hf_to_gguf.py D:\Chinese-LLaMA-Alpaca-3\lora-merge --outtype f16 --outfile D:\Chinese-LLaMA-Alpaca-3\lora-merge.gguf


最后看到以下日志说明执行成功

    INFO:hf-to-gguf:Model successfully exported to D:\llama-3-chinese-8b-instruct-lora-merge-gguf-quantize\llama-3-chinese-8B-instruct-lora-merge-F16.gguf


## 量化

进入 llama.cpp\build\bin\Release 目录，执行以下命令进行 gguf 的模型量化


     .\llama-quantize.exe D:\Chinese-LLaMA-Alpaca-3\lora-merge.gguf D:\Chinese-LLaMA-Alpaca-3\lora-merge-quantize.gguf q4_0  
 出现以下内容说明量化成功，这里生成了量化后的模型，大小从15G 压缩到了 4G 。
 
 

![image.png](https://p0-xtjj-private.juejin.cn/tos-cn-i-73owjymdk6/453a5fe5ccd647d293f315ba59b3373b~tplv-73owjymdk6-jj-mark-v1:0:0:0:0:5o6Y6YeR5oqA5pyv56S-5Yy6IEAg5oiR5piv546L5aSn5L2g5piv6LCB:q75.awebp?policy=eyJ2bSI6MywidWlkIjoiNTM2MjE3NDA1ODk1MTQ5In0%3D&rk3s=e9ecf3d6&x-orig-authkey=f32326d3454f2ac7e96d3d06cdbb035152127018&x-orig-expires=1731376969&x-orig-sign=fYbE2BxzwwBC850eHAzjaz7dj8A%3D)


如果你有自己的 gguf 模型，也可以直接用来量化，或者你懒得量化，可以直接从网上直接下载别人量化好的模型。

如果在量化的时候报错，可能是 peft 的版本太低了，升级到 0.7.1 以上，然后重新进行训练。

# ollama 部署
确保已经成功安装好了 ollama 。

创建 Modelfile ，写入下面内容，需要注意的是更换第一行量化模型 gguf 的位置，后面是提示模板和一些参数。


    FROM D:\Chinese-LLaMA-Alpaca-3\lora-merge-quantize.gguf
    TEMPLATE """{{ if .System }}<|start_header_id|>system<|end_header_id|>

    {{ .System }}<|eot_id|>{{ end }}{{ if .Prompt }}<|start_header_id|>user<|end_header_id|>

    {{ .Prompt }}<|eot_id|>{{ end }}<|start_header_id|>assistant<|end_header_id|>

    {{ .Response }}<|eot_id|>"""
    SYSTEM """你是一个助手。 You are a assistant."""
    PARAMETER temperature 0.2
    PARAMETER num_keep 24
    PARAMETER stop "<|start_header_id|>"
    PARAMETER stop "<|end_header_id|>"
    PARAMETER stop "<|eot_id|>"
    PARAMETER stop "<|reserved_special_token|>"

然后执行命令，输出 success 表示成功。

    ollama create my_llama3_q4 -f Modelfile

查看 ollama list 会看到有我们刚才创建的模型。然后执行 ollama run my_llama3_q4 即可进行对话，因为这里只是为了跑通微调、量化、部署整个流程，所以模型效果不好，所以我就不展示了。


 
