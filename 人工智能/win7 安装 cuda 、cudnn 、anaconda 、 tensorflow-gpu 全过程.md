## 前提
*     windows 7 操作系统
*     	已经安装好了 anaconda ，我的是 Anaconda3-2021.05-Windows-x86_64 
*     我的 GPU 是 NVIDIA GTX 1080 ，通过命令行 nvidia-smi 可以查看到 CUDA Version 为 10.0
*     根据[官网](https://tensorflow.google.cn/install/source_windows?hl=en#gpu)可以看到下图，我们准备安装 tensorflow-gpu-1.15.0 ，所以我们需要满足 cnDNN 版本为 7.4 ， CUDA 版本为 10.0 （我的显卡刚好满足），python3.5-3.7 （创建虚拟环境时候的 python 版本，目前经过实际测试想要安装 tensorflow-gpu-1.15.0 最高 python 版本为 3.6 ）。

![](https://img-blog.csdnimg.cn/20210816114108163.png?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L1JlZ2VlZWVlZQ==,size_16,color_FFFFFF,t_70)

## 搞定 CUDA 

* 进入 https://developer.nvidia.cn/cuda-toolkit-archive ，找到 CUDA Toolkit 10.0 ，因为我们的显卡 CUDA Version 为 10.0 ，然后选择 “windows”，“x86_64”，“10”，“exe（local）”进行下载即可
* 下载完成，双击进行安装，直接傻瓜式点下一步就好，直接往下，但是这里报错了，很多插件都未安装，经过在网上找答案，解决办法就是，在安装过程中，选择“自定义安装”，然后将 CUDA 下面的 Visual Studio Integration 这个去掉 ，再进行安装即可顺利完成。

## 搞定 CUDNN

* 进入 https://developer.nvidia.cn/rdp/cudnn-archive （如果无法访问，将 cn 变为 com ），找到 Download cuDNN v7.4.1 (Nov 8, 2018), for CUDA 10.0 版本 ，选择 cuDNN Library for Windows 7 版本 进行下载，一切都已自己的配置为准，我的配置在一开始已经说了。
* 这里需要账号，得注册一个登陆以后才能下载 ，总之要填写一堆东西
* 下载之后将压缩包解压，将 cuda 文件夹中


		（1）bin 文件夹里面的所有文件
		（2）include 文件夹里面的所有文件
		（3） lib\x64 文件夹所有文件

	对应复制到

		（1）C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
		（2）C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include
		（3）C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64
		
* 将下面的路径都加入到系统变量 PATH 中，可能部分在上面安装 CUDA 过程中已经有了，不要搞重复即可：

		C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\extras\CUPTI\lib64
		C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\include
		C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\lib\x64
		C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\bin
		C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v10.0\libnvvp




## 安装 anaconda 和配置环境变量

直接进去下面的网址找符和自己操作系统的版本 。

	https://www.anaconda.com/products/distribution#Downloads 

我安装的是历史版本 Anaconda3-2021.05-Windows-x86_64 ，傻瓜式下一步即可。


在环境变量中的系统变量 PATH 中加入自己的 anaconda 下的 Scripts 目录的路径，然后保存之后，重新打开命令行输入 conda ，如果有信息打印说明正常了。

## anaconda 安装 tensorflow-gpu


1.安装和配置 anaconda 完成之后 ， 命令行中 conda create -n tf\_gpu_1.15.0 python=3.6 ，过程中可能需要安装需要的包，结束之后可使用conda env list 查看当前虚拟环境列表，如果有了说明创建成功

2.命令行中 activate tf\_gpu_1.15.0 进入项目

3.命令行中 pip install tensorflow-gpu==1.15.0 ，时间比较长，显示 successfully ，表示安装成功 tensorflow-gpu==1.15.0 ，将项目需要的包都装好。


4.退出 tensorflow-gpu 虚拟环境 deactivate tf\_gpu_1.15.0 



## 用 Pycharm 使用 anaconda 虚拟环境

用 pycharm 创建新项目，setting —》project interpreter —》  Project Interpreter -》 Conda Environment -》 Existing environment -》找到虚拟环境的 python.exe 。如我的：C:\Users\QJFY-VR\anaconda3\envs\tf\_gpu_1.15.0\python.exe


运行以下代码

    import tensor flow as tf
    import os
    print('GPU',tf.test.is_gpu_available())
    a = tf.constant(2.)
    b = tf.constant(4.)
    print(a * b)

打印出“GPU True”,即代表GPU版本安装成功！

## jupyter 使用虚拟环境
现在已经将  tensorflow-gpu 安装到了名叫  tensorflow-gpu 的虚拟环境，为了能让 jupyter notebook 能够使用虚拟环境中的 tensorflow-gpu ，做法如下：

1.激活自己的虚拟环境 activate tensorflow-gpu 

2.安装 ipykernel  

        pip install ipykernel
        
   然后执行：
   
        python -m ipykernel install —name tensorflow-gpu 
        
3.打开 anaconda 的 jupyter notebook ，此时点击 “New” 创建文件会发现有自己的虚拟环境可选。


## 补充
这里有我安装过程中踩坑的详细记录，有需要自取即可。

* [《anaconda 创建虚拟环境时报错 HTTP errors 解决办法》](https://juejin.cn/post/7091119849098723336/)
* [《解决安装 anaconda 时报错  failed to create menus》](https://juejin.cn/post/7091120411747811335/)
* [《解决安装 CUDA 10.0 报错未安装组件》](https://juejin.cn/post/7091121474001436686/)
* [《解决 could not load dynamic library cudart64_100.dll》](https://juejin.cn/post/7091120966540984327/)




## 参考

https://blog.csdn.net/Regeeeeee/article/details/119714613
 