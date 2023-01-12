leetcode  2076. Process Restricted Friend Requests（python）

### 描述

You are given an integer n indicating the number of people in a network. Each person is labeled from 0 to n - 1.

You are also given a 0-indexed 2D integer array restrictions, where restrictions[i] = [x<sub>i</sub>, y<sub>i</sub>] means that person x<sub>i</sub> and person y<sub>i</sub> cannot become friends, either directly or indirectly through other people.

Initially, no one is friends with each other. You are given a list of friend requests as a 0-indexed 2D integer array requests, where requests[j] = [u<sub>j</sub>, v<sub>j</sub>] is a friend request between person u<sub>j</sub> and person v<sub>j</sub>.

A friend request is successful if u<sub>j</sub> and v<sub>j</sub> can be friends. Each friend request is processed in the given order (i.e., requests[j] occurs before requests[j + 1]), and upon a successful request, u<sub>j</sub> and v<sub>j</sub> become direct friends for all future friend requests.

Return a boolean array result, where each result[j] is true if the j<sub>th</sub> friend request is successful or false if it is not.

Note: If u<sub>j</sub> and v<sub>j</sub> are already direct friends, the request is still successful.

 




Example 1:


	Input: n = 3, restrictions = [[0,1]], requests = [[0,2],[2,1]]
	Output: [true,false]
	Explanation:
	Request 0: Person 0 and person 2 can be friends, so they become direct friends. 
	Request 1: Person 2 and person 1 cannot be friends since person 0 and person 1 would be indirect friends (1--2--0).
	
Example 2:


	Input: n = 3, restrictions = [[0,1]], requests = [[1,2],[0,2]]
	Output: [true,false]
	Explanation:
	Request 0: Person 1 and person 2 can be friends, so they become direct friends.
	Request 1: Person 0 and person 2 cannot be friends since person 0 and person 1 would be indirect friends (0--2--1).

Example 3:

	Input: n = 5, restrictions = [[0,1],[1,2],[2,3]], requests = [[0,4],[1,2],[3,1],[3,4]]
	Output: [true,false,true,false]
	Explanation:
	Request 0: Person 0 and person 4 can be friends, so they become direct friends.
	Request 1: Person 1 and person 2 cannot be friends since they are directly restricted.
	Request 2: Person 3 and person 1 can be friends, so they become direct friends.
	Request 3: Person 3 and person 4 cannot be friends since person 0 and person 1 would be indirect friends (0--4--3--1).


	

Note:

* 	2 <= n <= 1000
* 	0 <= restrictions.length <= 1000
* 	restrictions[i].length == 2
* 	0 <= x<sub>i</sub>, y<sub>i</sub> <= n - 1
* 	x<sub>i</sub> != y<sub>i</sub>
* 	1 <= requests.length <= 1000
* 	requests[j].length == 2
* 	0 <= u<sub>j</sub>, v<sub>j</sub> <= n - 1
* 	u<sub>j</sub> != v<sub>j</sub>


### 解析


根据题意，给定一个 n 个人。 每个人都被标记为从 0 到 n - 1。还给出了一个 0 索引的二维整数数组 restrictions ，其中 restrictions[i] = [x<sub>i</sub>, y<sub>i</sub>] 表示人 x<sub>i</sub> 和人 y<sub>i</sub> 不能直接或通过其他人间接成为朋友。

一开始互相都不是朋友。 给定一个好友请求列表，它是一个从 0 开始索引的二维整数数组 requests ，其中 requests[j] = [u<sub>j</sub>, v<sub>j</sub>] 是人 u<sub>j</sub> 和人 v<sub>j</sub> 之间的好友请求。如果 u<sub>j</sub> 和 v<sub>j</sub> 可以成为好友，则好友请求成功。 每个好友请求都按照给定的顺序进行处理（即 requests[j] 出现在 requests[j + 1] 之前），并且在请求成功后，u<sub>j</sub> 和 v<sub>j</sub> 成为所有未来好友请求的直接好友。

返回一个布尔数组结果，如果第 j 个好友请求成功，则每个 result[j] 为 True ，否则为 False。如果 u<sub>j</sub> 和 v<sub>j</sub> 已经是直接好友，则请求仍然成功。

读完题意其实很明显，最关键的地方 x 如果和 y 为敌，那么 x 和 y 的朋友都不能成为朋友。即 x 所在的朋友圈和 y 所在的朋友圈不能有交集，或者换一个思路，如果 x 所在的朋友圈和 y 所在的朋友圈中有存在于 restrictions 的朋友对，那么 x 和 y 也无法做朋友。最后判断每个 request 中两个人能否成为朋友，如果使用暴力解法需要 O(n^3) （因为要遍历 requests ，还有遍历 x 的朋友圈和 y 的朋友圈中的人组成的 pair 是否存在于 restrictions ），时间复杂度太高了，肯定会超时。换一个思路直接遍历 restrictions 中的某个对，只要其横跨 x 和 y 的朋友圈，则说明 x 和 y 无法成为朋友，反之则能成为朋友。

那么如何表示朋友圈呢，肯定不会是去找集合，那样的话又要遍历增加时间复杂度，而是找一个共同的圈子话事人，一般找的是数字最小或者数字最大的那个人，如果 request 的 x 和 y 两个的话事人为 X 和 Y ，和某个 restriction 的对 m 和 n 两个的话事人 M 和 N 一样，说明 x 和 y 无法成为朋友，否则可以成为朋友。

### 解答
				
	class Solution(object):
	    def friendRequests(self, n, restrictions, requests):
	        """
	        :type n: int
	        :type restrictions: List[List[int]]
	        :type requests: List[List[int]]
	        :rtype: List[bool]
	        """
	        father = {i:i for i in range(n)}
	        d = {i:[i] for i in range(n)}
	        result = []
	        for x,y in requests:
	            X = self.findFather(x, father)
	            Y = self.findFather(y, father)
	            if X==Y:
	                result.append(True)
	                continue
	            flag = True
	            for m,n in restrictions:
	                M = self.findFather(m, father)
	                N = self.findFather(n, father)
	                if (X==M and Y==N) or (X==N and Y==M):
	                    flag = False
	                    break
	            result.append(flag)
	            if flag:
	                self.union(x, y, father)
	        return result
	    
	    def union(self, x, y, father):
	        x = father[x]
	        y = father[y]
	        if x>y:
	            father[x] = y
	        elif x<y:
	            father[y] = x
	        
	        
	    def findFather(self, x, father):
	        if father[x] != x:
	            father[x] = self.findFather(father[x], father)
	        return father[x]
	            
	    
	                        
	            

            	      
			
### 运行结果

	Runtime: 5864 ms, faster than 48.27% of Python online submissions for Process Restricted Friend Requests.
	Memory Usage: 14.2 MB, less than 44.83% of Python online submissions for Process Restricted Friend Requests.



原题链接：https://leetcode.com/problems/process-restricted-friend-requests/



您的支持是我最大的动力
