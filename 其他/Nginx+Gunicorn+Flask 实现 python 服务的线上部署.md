# 前言
单纯使用 flask 提供服务会有很多问题， flask 官方也不建议这样做，所以想在线上生产环境部署 flask 服务需要一套完整的工具框架，也就是 nginx + gunicorn + flask 来完成。

# flask 
Flask 是一个轻量级的 Python Web 应用程序框架，其主要作用是帮助开发人员快速构建 Web 应用程序和 RESTful 服务。Flask 的优点就是上手快，简单易用，但是缺点就是这是一个同步框架，处理请求以单进程方式进行，一旦请求太多就容易崩掉，不适合线上使用，需要搭配其他工具使用。我们这里为了测试只是写了一个很简单的服务，逻辑就是睡眠一下然后打印一个包含了简单加法运算的字符串，很简单。

    # test.py
    import time
    from flask import Flask
    app = Flask(__name__)
    @app.route('/hello')
    def hello():
        time.sleep(0.001)
        return f'我要吃{10+10*10}个苹果'

    if __name__ == '__main__':
        app.run(host='0.0.0.0', port=5555)
        
首先我们单纯启动该服务，也就是启动该程序，然后使用下面的语句进行访问：

    curl http://你的主机ip:5555/hello
   
打印信息如下，表示服务能正常运行：

    我要吃110个苹果       
# gunicorn

Gunicorn（Green Unicorn）是一个 Python WSGI HTTP 服务器，它可以用来运行 Python Web 应用程序（如flask）。它的主要作用是充当一个中间层，用于处理客户端（如浏览器）和 Web 应用程序之间的通信，并确保请求能够高效地被传递给相应的 Python Web 应用程序。以下是 Gunicorn 的一些主要优点：

1.  **高性能**：Gunicorn 可以处理多个请求，可以通过多个工作进程并行处理请求，从而提高 Web 应用程序的处理能力，充分利用多核处理器,并且在高负载下表现出色，因此非常适合生产环境中的部署。
1.  **稳定性**：它具有稳定性和健壮性，可以处理高并发的请求，且很少会因为请求过载而崩溃。
1.  **部署简单**：Gunicorn 的安装和部署非常简单，可以与其他常用的 Web 服务器如 Nginx 配合使用，提供一个强大的 Web 应用程序部署方案。 

使用下面的命令进行安装：

    pip install gunicorn


安装完成之后直接在终端中找不到，因为这只是一个 python 包，所以我们要修改一个环境配置变量文件 /etc/profile ，在最后一行加入以下配置：

    export PATH=$PATH:/usr/local/Anaconda3/envs/tf-1.15.0-py3.6/bin

保存后执行 source /etc/profile 激活配置，命令行执行 gunicorn -v 可以看到版本号表示配置成功。新建一个文件 gunicorn.py ，如下：

    # 并行工作进程数
    workers = 16
    # 监听内网端口 5555
    bind = '你的主机ip:5555'
    # 设置访问日志和错误信息日志路径
    accesslog = '/tmp/gunicorn_acess.log'
    errorlog = '/tmp/gunicorn_error.log'
    # 设置日志记录水平
    loglevel = 'warning'

启动 gunicorn ，命令如下，test 就是 python 文件名，app 就是 flask 的实例名：

    gunicorn -c gunicorn.py test:app

使用和上面同样的命令进行访问，会打印相同的日志结果信息：

    curl http://你的主机ip:5555/hello


# Apache Benchmark（简称 ab ）

Apache Benchmark（ab）是一个用于 Apache HTTP 服务器的轻量级命令行工具，用于执行基准测试或压力测试，以评估 Web 服务器的性能。它的主要作用是模拟多个并发用户向服务器发送请求，以测量服务器在不同负载下的性能表现。最主要的是上手快，而且免费。使用以下命令进行安装：

    yum -y install httpd-tools

安装结束之后，通过命令 ab -v 可以看到版本号表示安装成功。

# flask VS gunicorn

我们使用单纯的 flask 服务进行测试，命令如下，表示模拟 100 个用户发起 10000 个请求：

    ab -n 10000 -c 100 你的主机ip:5555/hello

结果如下，可以看到，单进程每秒处理请求 790.35 条：

    Completed 1000 requests
    Completed 2000 requests
    Completed 3000 requests
    Completed 4000 requests
    Completed 5000 requests
    Completed 6000 requests
    Completed 7000 requests
    Completed 8000 requests
    Completed 9000 requests
    Completed 10000 requests
    Finished 10000 requests


    Server Software:        Werkzeug/2.0.3
    Server Hostname:        你的主机ip
    Server Port:            5555

    Document Path:          /hello
    Document Length:        21 bytes

    Concurrency Level:      100
    Time taken for tests:   12.653 seconds
    Complete requests:      10000
    Failed requests:        0
    Write errors:           0
    Total transferred:      1750000 bytes
    HTML transferred:       210000 bytes
    Requests per second:    790.35 [#/sec] (mean)
    Time per request:       126.527 [ms] (mean)
    Time per request:       1.265 [ms] (mean, across all concurrent requests)
    Transfer rate:          135.07 [Kbytes/sec] received

    ......
    
服务器资源占用：


![5d9c8c83ef6f4fc28af11820ae2503a.png](https://p6-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/0a075797751b4653905088efe5c18a0d~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=533&h=72&s=5958&e=png&b=000000)

同样的请求操作，我们启动 gunicorn 来进行测试，日志打印如下，可以看到，每秒处理请求 7593.84 条，提升近 10 倍，足可以看出来 gunicorn 的强大之处：

    Completed 1000 requests
    Completed 2000 requests
    Completed 3000 requests
    Completed 4000 requests
    Completed 5000 requests
    Completed 6000 requests
    Completed 7000 requests
    Completed 8000 requests
    Completed 9000 requests
    Completed 10000 requests
    Finished 10000 requests


    Server Software:        gunicorn
    Server Hostname:        你的主机ip
    Server Port:            5555

    Document Path:          /hello
    Document Length:        21 bytes

    Concurrency Level:      100
    Time taken for tests:   1.317 seconds
    Complete requests:      10000
    Failed requests:        0
    Write errors:           0
    Total transferred:      1740000 bytes
    HTML transferred:       210000 bytes
    Requests per second:    7593.84 [#/sec] (mean)
    Time per request:       13.169 [ms] (mean)
    Time per request:       0.132 [ms] (mean, across all concurrent requests)
    Transfer rate:          1290.36 [Kbytes/sec] received

    ......

服务器资源占用：

![39ec0bfe909207b36d1e5a25dc20124.png](https://p3-juejin.byteimg.com/tos-cn-i-k3u1fbpfcp/f40a6279097d4cf8a1324760a1958786~tplv-k3u1fbpfcp-jj-mark:0:0:0:0:q75.image#?w=540&h=261&s=23561&e=png&b=000000)
# nginx

Nginx 是一个高性能的开源 Web 服务器，也可用作反向代理服务器、负载均衡器和 HTTP 缓存。其主要作用包括： Nginx 的优点包括：

1.  **高性能**：Nginx 具有卓越的性能和处理高并发连接的能力，适合在高流量网站和应用程序中使用。作为反向代理服务器和负载均衡器时表现出色，能够平衡流量并确保服务器的稳定性和可靠性。
1.  **低内存消耗**：Nginx 使用内存较少，能够处理大量的并发请求，这使得它在资源受限的环境中具有优势。
1.  **可扩展性**：Nginx 的架构设计允许它轻松地扩展以适应不断增长的需求，可根据需要添加更多的服务器和节点。
1.  **灵活性**：Nginx 的配置灵活，可以根据需求进行高度定制，能够满足各种不同类型的 Web 应用程序的需求。
 
配置如下：

    upstream flask{
        server 你的主机ip:5555;
    }

    server {
        listen 8080;
        server_name localhost; 
        # 自定义的前缀
        location /gunicorn {
            # 请求转发到 gunicorn 服务，同时启动了很多的 flask 进程
            proxy_pass http://flask; 
            # 设置请求头，并将头信息传递给服务器端 
            proxy_set_header Host $host; 
        }
    }
    
使用如下命令访问：

    curl http://nginx服务器ip:8080/gunicorn/hello
    
打印如下，其实和上面一样：

    我要吃110个苹果
    
# 参考

- https://www.zhihu.com/question/38528616
- https://bbs.huaweicloud.com/blogs/309794
- https://www.cnblogs.com/yanshw/p/11208964.html
- https://juejin.cn/post/6844904053755871239?from=search-suggest
- https://codeantenna.com/a/WK66T0ZmRC
- https://juejin.cn/post/6844904009157836808