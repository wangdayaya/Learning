### 前言

我们平时在写 python 项目的时候，尤其是深度学习项目，可能会配置很多超参数，管理起来很混乱，这时我们就可以把配置都写入一个 yml 文件，然后使用这个 hb-config 库轻松管理。

先上链接，以示尊重：https://github.com/hb-research/hb-config 。看作者一句话对库功能进行总结：

- hb-config is utility for easy to configure your python project especially **Deep Learning experiments**.

### 安装

安装步骤很简单，直接调用 pip 命令即可

	$ pip install hb-config
	
### 使用

首先必须要有一个名为 myConfig 的文件夹，然后里面假如我们有一个 config.yml 配置文件，不懂写 yml 文件的，自己百度一下，五分钟就可以学会。
	
	project: "hb-config"
	example: true
	people:
	  name: "wangda"
	  sex:  "male"
	  
使用 jupyter ，调用库中的 Config ，将文件名传入：

	from hbconfig import Config
	Config("myConfig/config.yml")
	print(Config)
	
结果打印：

	Read config file name: myConfig/config
	{
	    "project": "hb-config",
	    "example": true,
	    "people": {
	        "name": "wangda",
	        "sex": "male"
	    }
	}
	
需要注意的是这里可能会因为 PyYAML 库版本过高而报错，降到 5.4.1 就可以了。

### 取值

1. 取 project 值：


		print(Config.project)



	
	结果打印：
	
		hb-config
	
2. 取 example 值：

	
		print(Config. example)
		
	结果打印：
	
		True
	
3. 取人的名字：

		print(Config.people.name)
	
	结果打印：
	
		wangda
	
4. 取人这个对象：


		print(Config.people)
	
	结果打印：
	
		{
		    "name": "wangda",
		    "sex": "male",
		    "get_tag": "people"
		}	

### 增加新参数

我们可以直接使用 Config 设置新的属性及其值，但是不改变配置文件
	
	Config.height = 1.8
	print(Config.height)
结果打印：
	
	1.8
再次输出 Config 
	
	print(Config)
结果打印，可以看出没有发生变化：
	
	Read config file name: myConfig/config
	{
	    "project": "hb-config",
	    "example": true,
	    "people": {
	        "name": "wangda",
	        "sex": "male"
	    }
	}
	