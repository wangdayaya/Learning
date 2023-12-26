

我们安装的时候，直接进去下面的网址找符和自己操作系统的版本 。

	https://www.anaconda.com/products/distribution#Downloads 

这里需要注意的是如果选择 anaconda 版本太新，当前最新 Anaconda3-2021.11-Windows-x86_64.exe ，可能在 win7 系统上面安装的时候会报错，比如我的报错为 failed to create menus ，而详情原因是找不到模块：

	 C:\Users\QJFY\AppData\Local\Temp\_MEI99282\python39.dll 
	
（WTF）这报错莫名其妙的，只能归结为版本和系统不兼容了，然后去历史版本的网页中

	https://repo.anaconda.com/archive/

找次新的 anaconda 版本安装试试，我降了一个版本，换成了 Anaconda3-2021.05-Windows-x86_64.exe 安装一切顺利。