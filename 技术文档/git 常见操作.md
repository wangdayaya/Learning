查看远程 git 地址：

	git remote -v

删除远程项目地址：

	git remote rm origin

添加远程代码仓库：

	git remote add origin http://10.5.1.22:7990/scm/address/spatial-semantic-v2-be-engine.git

已有 git 远程仓库地址情况下，修改成新的 git 远程仓库地址： 

	git remote set-url origin http://10.5.1.22:7990/scm/address/spatial-semantic-v2-be-engine.git
	
克隆代码：

	git clone http://10.5.1.22:7990/scm/address/spatial-semantic-v2-be-engine.git
	
拉取代码：

	git pull origin 分支名
	

更新代码：

	git status  # 查看文件状态
	git add .
	git commit -m “说明”
	git push
	
新建本地分支：

	git branch 分支名
	
删除本地分支：

	git branch -d 分支名
	
删除远程分支： 

	git push origin -d 分支名

查看本地分支：
	
	git branch
	
查看远程分支：

	git branch -a
	
切换分支：

	git checkout 分支名

拉取远程仓库里面的其他分支

	 git checkout -b 1215xcswGIS origin/1215xcswGIS   # 作用是 checkout 远程的 dev 分支，在本地起名为 dev 分支，并切换到本地的 dev 分支
	 
git pull 是上下文环境敏感的，它会把所有的提交自动给你并 merge 到当前分支当中，没有复查的过程

而 git fetch 只是把拉去的提交存储到本地仓库中，真正合并到主分支中需要使用 merage
