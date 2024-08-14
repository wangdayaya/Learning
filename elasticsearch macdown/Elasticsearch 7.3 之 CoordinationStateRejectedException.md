## Elasticsearch 7.3 之 CoordinationStateRejectedException

集群中的某一个数据节点关闭之后重启报出如下异常，但是没有影响正常的启动：

	Caused by: org.elasticsearch.cluster.coordination.CoordinationStateRejectedException: incoming term 19 does not match current term 20
		at org.elasticsearch.cluster.coordination.CoordinationState.handleJoin(CoordinationState.java:223) ~[elasticsearch-7.3.0.jar:7.3.0]
		at org.elasticsearch.cluster.coordination.Coordinator.handleJoin(Coordinator.java:948) ~[elasticsearch-7.3.0.jar:7.3.0]
		at java.util.Optional.ifPresent(Optional.java:159) ~[?:1.8.0_261]
		at org.elasticsearch.cluster.coordination.Coordinator.processJoinRequest(Coordinator.java:517) ~[elasticsearch-7.3.0.jar:7.3.0]
		at org.elasticsearch.cluster.coordination.Coordinator.handleJoinRequest(Coordinator.java:484) ~[elasticsearch-7.3.0.jar:7.3.0]
		at org.elasticsearch.cluster.coordination.JoinHelper.lambda$new$0(JoinHelper.java:125) ~[elasticsearch-7.3.0.jar:7.3.0]
		at org.elasticsearch.xpack.security.transport.SecurityServerTransportInterceptor$ProfileSecuredRequestHandler$1.doRun(SecurityServerTransportInterceptor.java:257) [x-pack-security-7.3.0.jar:7.3.0]
		at org.elasticsearch.common.util.concurrent.AbstractRunnable.run(AbstractRunnable.java:37) [elasticsearch-7.3.0.jar:7.3.0]
		at org.elasticsearch.xpack.security.transport.SecurityServerTransportInterceptor$ProfileSecuredRequestHandler.messageReceived(SecurityServerTransportInterceptor.java:315) [x-pack-security-7.3.0.jar:7.3.0]
		at org.elasticsearch.transport.RequestHandlerRegistry.processMessageReceived(RequestHandlerRegistry.java:63) [elasticsearch-7.3.0.jar:7.3.0]
		at org.elasticsearch.transport.TransportService$7.doRun(TransportService.java:703) [elasticsearch-7.3.0.jar:7.3.0]
		at org.elasticsearch.common.util.concurrent.ThreadContext$ContextPreservingAbstractRunnable.doRun(ThreadContext.java:758) [elasticsearch-7.3.0.jar:7.3.0]
		at org.elasticsearch.common.util.concurrent.AbstractRunnable.run(AbstractRunnable.java:37) [elasticsearch-7.3.0.jar:7.3.0]
		at java.util.concurrent.ThreadPoolExecutor.runWorker(ThreadPoolExecutor.java:1149) [?:1.8.0_261]
		at java.util.concurrent.ThreadPoolExecutor$Worker.run(ThreadPoolExecutor.java:624) [?:1.8.0_261]
		at java.lang.Thread.run(Thread.java:748) [?:1.8.0_261]

参考解释：网上有人爆此异常无法启动，搜索说是因为该节点之前运行过 es 实例已经创建了 data 目录，如果与要加入的集群可能存在冲突，影响正常启动，最后解决方法是删除 data 文件夹
