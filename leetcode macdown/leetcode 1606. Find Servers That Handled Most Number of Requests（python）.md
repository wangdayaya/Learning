leetcode  1606. Find Servers That Handled Most Number of Requests（python）

### 描述


You have k servers numbered from 0 to k-1 that are being used to handle multiple requests simultaneously. Each server has infinite computational capacity but cannot handle more than one request at a time. The requests are assigned to servers according to a specific algorithm:

* The i<sup>th</sup> (0-indexed) request arrives.
* If all servers are busy, the request is dropped (not handled at all).
* If the (i % k)<sup>th</sup> server is available, assign the request to that server.
* Otherwise, assign the request to the next available server (wrapping around the list of servers and starting from 0 if necessary). For example, if the i<sup>th</sup> server is busy, try to assign the request to the (i+1)<sup>th</sup> server, then the (i+2)<sup>th</sup> server, and so on.

You are given a strictly increasing array arrival of positive integers, where arrival[i] represents the arrival time of the i<sup>th</sup> request, and another array load, where load[i] represents the load of the i<sup>th</sup> request (the time it takes to complete). Your goal is to find the busiest server(s). A server is considered busiest if it handled the most number of requests successfully among all the servers.

Return a list containing the IDs (0-indexed) of the busiest server(s). You may return the IDs in any order.


Example 1:


![](https://assets.leetcode.com/uploads/2020/09/08/load-1.png)
	
	Input: k = 3, arrival = [1,2,3,4,5], load = [5,2,3,3,3] 
	Output: [1] 
	Explanation:
	All of the servers start out available.
	The first 3 requests are handled by the first 3 servers in order.
	Request 3 comes in. Server 0 is busy, so it's assigned to the next available server, which is 1.
	Request 4 comes in. It cannot be handled since all servers are busy, so it is dropped.
	Servers 0 and 2 handled one request each, while server 1 handled two requests. Hence server 1 is the busiest server.
	
Example 2:

	Input: k = 3, arrival = [1,2,3,4], load = [1,2,1,2]
	Output: [0]
	Explanation:
	The first 3 requests are handled by first 3 servers.
	Request 3 comes in. It is handled by server 0 since the server is available.
	Server 0 handled two requests, while servers 1 and 2 handled one request each. Hence server 0 is the busiest server.


Example 3:


	Input: k = 3, arrival = [1,2,3], load = [10,12,11]
	Output: [0,1,2]
	Explanation: Each server handles a single request, so they are all considered the busiest.
	
Example 4:

	Input: k = 3, arrival = [1,2,3,4,8,9,10], load = [5,2,10,3,1,2,2]
	Output: [1]

	
Example 5:

	Input: k = 1, arrival = [1], load = [1]
	Output: [0]


Note:

	1 <= k <= 10^5
	1 <= arrival.length, load.length <= 10^5
	arrival.length == load.length
	1 <= arrival[i], load[i] <= 10^9
	arrival is strictly increasing.


### 解析

根据题意，有 k 个服务器，编号从 0 到 k-1 ，用于同时处理多个请求。每个服务器都有无限的计算能力，但一次不能处理多个请求。根据特定算法将请求分配给服务器：

* 第 i 个（ 从 0 开始索引）请求到达
* 如果所有服务器都忙，则请求将被丢弃（根本不处理）
* 如果第 (i % k) 个服务器可用，则将请求分配给该服务器
* 否则，将请求分配给下一个可用的服务器（环绕服务器列表并在必要时从 0 开始），例如，如果第 i 个服务器繁忙，则尝试将请求分配给第 (i+1) 个服务器，然后是第 (i+2) 个服务器，依此类推。

给定一个严格递增的正整数数组 arrival ，其中 arrival[i] 表示第 i 个请求的到达时间，以及另一个数组 load ，其中 load[i] 表示第 i 个请求的负载（完成所需的时间）。目标是找到最繁忙的服务器。如果服务器在所有服务器中成功处理的请求数量最多，则该服务器被认为是最繁忙的。返回包含最繁忙服务器的 ID 的列表。可以按任何顺序返回 ID。

题目很繁杂，但是理解之后也比较简单，解法的主要思想是维护两个列表，一个是空闲服务器列表 free ，另一个是正在执行请求的服务器列表 buzy ，每次得到一个新的请求的时候，如果 free 为空，丢弃该请求即可，如果不为空从第 (i % k) 个服务器中向后找空闲服务器，如果没有则从第 0 个开始找空闲服务器。

有两个关键的地方需要注意，一个是初始化 free 和 buzy 的时候选用能自动排序的类，另一个是在 free 中找空闲服务器时最好有内置函数，如果没有则用二分法进行找，如果这两个步骤处理不好很容易超时，我的代码也是刚好过，耗时还是很严重。

### 解答
				

	from sortedcontainers import SortedList
	class Solution(object):
	    def busiestServers(self, k, arrival, load):
	        """
	        :type k: int
	        :type arrival: List[int]
	        :type load: List[int]
	        :rtype: List[int]
	        """
	        free = SortedList([i for i in range(k)])
	        buzy = SortedList([],key=lambda x:-x[1])
	        count = {i:0 for i in range(k)}
	        for i,start in enumerate(arrival):
	            while(buzy and buzy[-1][1]<=start):
	                pair = buzy.pop()
	                free.add(pair[0])
	            if not free: continue
	            id = self.findServer(free, i%k)
	            count[id] += 1
	            free.remove(id)
	            buzy.add([id, start+load[i]])
	        result = []
	        MAX = max(count.values())
	        for k,v in count.items():
	            if v == MAX:
	                result.append(k)
	        return result
	    def findServer(self, free, id):
	        idx = bisect.bisect_right(free, id-1)
	        if idx!=len(free):
	            return free[idx]
	        return free[0]
	            
	            
	            
	                    
			
### 运行结果

	Runtime: 5120 ms, faster than 10.00% of Python online submissions for Find Servers That Handled Most Number of Requests.
	Memory Usage: 38.6 MB, less than 50.00% of Python online submissions for Find Servers That Handled Most Number of Requests.



原题链接：https://leetcode.com/problems/find-servers-that-handled-most-number-of-requests/



您的支持是我最大的动力
