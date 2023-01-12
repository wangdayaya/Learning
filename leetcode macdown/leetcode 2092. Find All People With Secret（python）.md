leetcode  2092. Find All People With Secret（python）

### 描述


You are given an integer n indicating there are n people numbered from 0 to n - 1. You are also given a 0-indexed 2D integer array meetings where meetings[i] = [x<sub>i</sub>, y<sub>i</sub>, time<sub>i</sub>] indicates that person x<sub>i</sub> and person y<sub>i</sub> have a meeting at time<sub>i</sub>. A person may attend multiple meetings at the same time. Finally, you are given an integer firstPerson.

Person 0 has a secret and initially shares the secret with a person firstPerson at time 0. This secret is then shared every time a meeting takes place with a person that has the secret. More formally, for every meeting, if a person x<sub>i</sub> has the secret at time<sub>i</sub>, then they will share the secret with person y<sub>i</sub>, and vice versa.

The secrets are shared instantaneously. That is, a person may receive the secret and share it with people in other meetings within the same time frame.

Return a list of all the people that have the secret after all the meetings have taken place. You may return the answer in any order.




Example 1:

	Input: n = 6, meetings = [[1,2,5],[2,3,8],[1,5,10]], firstPerson = 1
	Output: [0,1,2,3,5]
	Explanation:
	At time 0, person 0 shares the secret with person 1.
	At time 5, person 1 shares the secret with person 2.
	At time 8, person 2 shares the secret with person 3.
	At time 10, person 1 shares the secret with person 5.​​​​
	Thus, people 0, 1, 2, 3, and 5 know the secret after all the meetings.

	
Example 2:


	Input: n = 4, meetings = [[3,1,3],[1,2,2],[0,3,3]], firstPerson = 3
	Output: [0,1,3]
	Explanation:
	At time 0, person 0 shares the secret with person 3.
	At time 2, neither person 1 nor person 2 know the secret.
	At time 3, person 3 shares the secret with person 0 and person 1.
	Thus, people 0, 1, and 3 know the secret after all the meetings.

Example 3:


	Input: n = 5, meetings = [[3,4,2],[1,2,1],[2,3,1]], firstPerson = 1
	Output: [0,1,2,3,4]
	Explanation:
	At time 0, person 0 shares the secret with person 1.
	At time 1, person 1 shares the secret with person 2, and person 2 shares the secret with person 3.
	Note that person 2 can share the secret at the same time as receiving it.
	At time 2, person 3 shares the secret with person 4.
	Thus, people 0, 1, 2, 3, and 4 know the secret after all the meetings.
	
Example 4:

	
	Input: n = 6, meetings = [[0,2,1],[1,3,1],[4,5,1]], firstPerson = 1
	Output: [0,1,2,3]
	Explanation:
	At time 0, person 0 shares the secret with person 1.
	At time 1, person 0 shares the secret with person 2, and person 1 shares the secret with person 3.
	Thus, people 0, 1, 2, and 3 know the secret after all the meetings.
	


Note:


* 2 <= n <= 10^5
* 1 <= meetings.length <= 10^5
* meetings[i].length == 3
* 0 <= x<sub>i</sub>, y<sub>i</sub> <= n - 1
* x<sub>i</sub> != y<sub>i</sub>
* 1 <= time<sub>i</sub> <= 10^5
* 1 <= firstPerson <= n - 1

### 解析

根据题意，给定一个整数 n 表示有 n 个人，编号从 0 到 n - 1。给出一个 0 索引的二维整数数组 meetings ，其中 meetings[i] = [x<sub>i</sub>, y<sub>i</sub>, time<sub>i</sub>] 表示人 x<sub>i</sub> 和人 y<sub>i</sub>有一个会议在 time<sub>i</sub> 。 一个人可以同时参加多个会议。 还给出一个整数 firstPerson 。

第 0 个人有一个秘密，并且最初在 0 时与 firstPerson 共享该秘密。然后每次与拥有该秘密的人开会时都会共享该秘密。 更正式地，对于每次会议，如果某人 x<sub>i</sub> 在 time<sub>i</sub> 拥有秘密，那么他们将与人 y<sub>i</sub> 共享秘密，反之亦然。秘密是即时共享的。 也就是说，一个人可能会收到秘密并在同一时间范围内与其他会议中的人分享。题目要求在所有会议发生后，返回所有知道秘密的人的列表。，可以按任何顺序返回答案。

读完题我们肯定对题意了解比较清楚，对于秘密的知情者和不知情者的集合划分情况，我们肯定自然而然地第一时间想到用 union find 解题，关键在于同一时刻开会的人员的秘密共享解决方法设计，思路如下：

* 肯定要把 meetings 按照时间的顺序进行排序，因为先知道秘密的人可以在同一时刻或者后续时刻开会的时候分享给其他人，我们可以直接使用排序对 meetings 中的元素按照开会时间进行升序排序
* 对于同一时间开会的人，用到 union find 算法，将他们的祖先设置为开会人当中最小的数字，当然如果祖先为 0 ，说明这个人知道秘密
* 在同一时刻的所有开完会的人可能会分成两个集合，一个集合中的人会知道秘密，另一集合中的人还是不知道秘密。对于知道秘密的人我们知道其祖先为 0 ，并将其加入到结果集合 reuslt 中，如果不知道秘密那么其祖先肯定不为 0 ，那么我们将其祖先重置为他自己，因为在这个时刻两个人的会议祖先如果不为 0 那么不能用于之后的时刻
* 其中 union 和 findFather 两个函数是 union find 算法的通用模版，我们应该将其记住，到哪里都是一样的
* 最后只需要将结果集合 result 变为列表返回即可。

### 解答
				
	class Solution(object):
	    def findAllPeople(self, n, meetings, firstPerson):
	        """
	        :type n: int
	        :type meetings: List[List[int]]
	        :type firstPerson: int
	        :rtype: List[int]
	        """
	        meetings.sort(key=lambda x: x[2])
	        father = {i: i for i in range(n)}
	        father[firstPerson] = 0
	        result = set()
	        result.add(0)
	        result.add(firstPerson)
	        i = 0
	        while i < len(meetings):
	            j = i
	            persons = set()
	            while j < len(meetings) and meetings[j][2] == meetings[i][2]:
	                x = meetings[j][0]
	                y = meetings[j][1]
	                persons.add(x)
	                persons.add(y)
	                if self.findFather(x, father) != self.findFather(y, father):
	                    self.union(x, y, father)
	                j += 1
	
	            for p in persons:
	                if self.findFather(p, father) == 0 :
	                    result.add(p)
	                else:
	                    father[p] = p
	
	            i = j
	        return list(result)
	
	    def union(self, x, y, father):
	        x = father[x]
	        y = father[y]
	        if x > y:
	            father[x] = y
	        else:
	            father[y] = x
	
	    def findFather(self, x, father):
	        if father[x] != x:
	            father[x] = self.findFather(father[x], father)
	        return father[x]
	
	

            	      
			
### 运行结果

	
	Runtime: 2352 ms, faster than 59.22% of Python online submissions for Find All People With Secret.
	Memory Usage: 53.6 MB, less than 79.61% of Python online submissions for Find All People With Secret.

原题链接：https://leetcode.com/problems/find-all-people-with-secret/



您的支持是我最大的动力
