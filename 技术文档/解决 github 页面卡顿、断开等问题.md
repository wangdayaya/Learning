解决 github 页面卡顿、断开等问题

## mac 系统

1.打开 http://ipaddress.com 网页，找出下面三个网址对应的 ip ，我的结果如下：

	140.82.114.4 github.com 
	185.199.108.153 assets-cdn.github.com 
	199.232.69.194 github.global.ssl.fastly.net

2.使用超级用户权限修改 sudo vi /etc/hosts 文件（这个文件对一般用户是只读的），将上面的内容拷贝到文件末尾，然后保存文件。

3.执行 ping github.com 查看结果，此时不再会出现超时的现象：

	64 bytes from 140.82.114.4: icmp_seq=0 ttl=48 time=320.110 ms
	64 bytes from 140.82.114.4: icmp_seq=1 ttl=48 time=319.992 ms
	64 bytes from 140.82.114.4: icmp_seq=2 ttl=48 time=320.131 ms
	...
	
## windows 系统

1.第一步和上面的一样，找出三个网址对应的 ip ，我的结果如下：

	140.82.112.3 github.com
	185.199.108.153 assets-cdn.github.com
	199.232.69.194 github.global.ssl.fastly.net
	
2.修改 C:\Windows\System32\drivers\etc\hosts 的文件权限（一般的用户只有可读权限，无法进行修改，怎么改 hosts 权限？直接点击文件右键选择“属性”，然后在“安全”栏中给当前的用户赋与各种权限），然后将相同的内容拷贝到文件末尾，保存即可。

3.在终端中执行下面命令，刷新 DNS 解析缓存

	 ipconfig /flushdns

4.执行 ping github.com 查看结果，此时不再会出现超时的现象：

	来自 140.82.112.3 的回复：字节=32 时间=212 ms TTL=41
	来自 140.82.112.3 的回复：字节=32 时间=215 ms TTL=41
	...

## 总结

果然技术才是第一生产力，改了之后页面再也不卡顿了，尽情 cv 享用代码！