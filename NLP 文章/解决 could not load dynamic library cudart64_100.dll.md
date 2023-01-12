安装好 anaconda 、 cuda 、cudnn 以及 tensorflow 之后，并且我保证了他们的版本一致性，运行代码仍然报错无法加载 cudart64_100.dll 这个动态库，而且该文件就存在于我的 CUDA\bin 目录下，我的 CUDA 环境变量也都已经加到 PATH，这时我们只需要把这个文件文件复制到 C:\windows\system 中即可大功告成，运行打印：

	successfully opened dynamic libaray cudart64_100.dll
	
据说原因是因为 windows 就算添加了系统变量还是无法应用于 dll 命令文件。

如果运行过程中还有其他的 dll 文件无法加载，可以同样适用上面的解决方法，像我在操作过程中还需要拷贝的文件有：

	cublas64_100.dll
	cufft64_100.dll
	curand64_100.dll
	cusolver64_100.dll
	cusparse64_100.dll
	cudnn64_7.dll