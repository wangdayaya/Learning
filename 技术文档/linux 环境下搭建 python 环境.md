## linux 环境下搭建 python 环境

1. 内网无法访问外网，所以在 /etc/yum.conf 中最后一行加入下面一行配置 yum 代理
	
		proxy=http://10.5.1.101:3128/ 
2. yum 安装 python3.6  
   
		yum -y install epel-release (如果安装过则跳过) 
 		yum -y install python36u  
3. yum 安装了pip3        
		
		yum -y install epel-release (如果安装过则跳过) 
  		yum -y install python36u-pip
4. pip3 使用代理进行 pip3 升级      

		pip3 install —upgrade —proxy http://10.5.1.101:3128 pip
5. pip3 使用代理安装 tensorflow
		
		pip3 install —proxy http://10.5.1.101:3128 tensorflow-cpu==1.15
6. pip3 使用代理安装 flask    
		
		pip3 install —proxy http://10.5.1.101:3128 flask
7. pip3 使用代理安装 pandas    
		
		pip3 install —proxy http://10.5.1.101:3128 pandas
8. 后台运行 python 代码
		
		nohup /usr/bin/python3 findFields.py & 
 


