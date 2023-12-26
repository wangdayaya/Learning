# 前言
tensorflow > 2.10 的 cpu 和 gpu 版本不进行区分，里面既有 cpu 版本又有 gpu 和 tpu 版本，所以只需要简单装一个 tensorflow 即可，但是想在 window 中使用 tensorflow>2.10 的 gpu 版本必须要在 wsl 环境下进行环境的搭建，同时为了开发方便，使用 pycharm 专业版配置连接 wsl 环境中的虚拟环境即可。
 
# 准备
- window 10
- 显卡
- pycharm 专业版

# window10 安装 wsl
- win+S 搜索“启用或关闭 Windows 功能”，确保勾选住“适用于Linux 的 Windows 子系统”，并保存配置
- win+S 找到并使用管理员身份打开 powershell
- 执行 wsl --install
- 执行 wsl --set-default-version 2 调整版本为  WSL2 
- win+S 搜索 ubuntu 启动即可

# ubuntu 中安装环境

- 安装 anaconda
    -   执行 wget https://repo.anaconda.com/archive/Anaconda3-2023.07-2-Linux-x86_64.sh 下载安装包
    -   执行 bash Anaconda3-2023.07-2-Linux-x86_64.sh 开始安装，按照提示执行默认的傻瓜操作即可
    -   执行 source ~/.bashrc 整理环境
    -   执行 conda create -n tf-2.12-py-3.10 python=3.10 创建虚拟环境
    -   进入虚拟环境 conda activate tf-2.12-py-3.10
  
- 安装 cudatoolkit 、 cudnn 以此执行下面的命令

    ```
    conda install -c conda-forge cudatoolkit=11.8.0
    pip install nvidia-cudnn-cu11==8.6.0.163
    CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))
    export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH
    mkdir -p $CONDA_PREFIX/etc/conda/activate.d
    echo 'CUDNN_PATH=$(dirname $(python -c "import nvidia.cudnn;print(nvidia.cudnn.__file__)"))' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    echo 'export LD_LIBRARY_PATH=$CUDNN_PATH/lib:$CONDA_PREFIX/lib/:$LD_LIBRARY_PATH' >> $CONDA_PREFIX/etc/conda/activate.d/env_vars.sh
    ```
 

- 安装 tensorflow
    - pip install tensorflow==2.12.0

# 验证

完成上面的步骤之后，在 ubuntu 命令行中执行下面命令，看是否安装 CPU 版本成功：

```
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
```
结果打印：
```
tf.Tensor(295.5964, shape=(), dtype=float32)
```
在 ubuntu 命令行中执行下面命令，看是否能用 GPU 版本：

```
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
```
结果打印：
```
[PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
```
# pycharm 连接 ubuntu 中的虚拟环境

- 下载 pycharm 专业版
- settings -> python interpreter ，页面中找到下拉三角找到“ON WSL” ，然后找到 “Conda Environment” 中点击目录找到自己虚拟环境中的 python 位置保存即可

# 完结撒花

可以开心的在 window 上面使用 tensorflow 2.12 进行项目开发和训练了！

# 参考

https://pureinfotech.com/install-windows-subsystem-linux-2-windows-10/
https://hackmd.io/@Kailyn/H1lKxHKeF
https://www.tensorflow.org/install/pip