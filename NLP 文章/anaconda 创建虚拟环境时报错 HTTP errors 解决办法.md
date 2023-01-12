使用 anaconda 创建虚拟环境报错：

	CondaHTTPError: HTTP 000 CONNECTION FAILED for url <https://repo.anaconda.com/pkgs/main/win-64/current_repodata.json>
	
	 An HTTP error occurred when trying to retrieve this URL.
	HTTP errors are often intermittent, and a simple retry will get you on your way
	
如果是刚安装好的 anaconda ，用户目录下没有 .condarc 文件，在命令行中挨个输入以下命令：

	conda config --add channels https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
	conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
	conda config --add channels https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
	conda config --set show_channel_urls yes
	conda config --set ssl_vertify yes
	
此时在用户目录下面会出现一个文件 .condarc ，这时我们就可以直接在文件中修改了，因为命令行修改太麻烦了，将 -defaults 删除，此时文件内容就剩下：

	channels:
	  - https://mirrors.ustc.edu.cn/anaconda/pkgs/free/
	  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/cloud/conda-forge/
	  - https://mirrors.tuna.tsinghua.edu.cn/anaconda/pkgs/free/
	show_channel_urls: true
	ssl_verify: true

然后我们重新创建虚拟环境成功