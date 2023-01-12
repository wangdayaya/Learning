leetcode  997. Find the Town Judge（python）

### 描述


In a town, there are N people labelled from 1 to N.  There is a rumor that one of these people is secretly the town judge.

If the town judge exists, then:

* The town judge trusts nobody.
* Everybody (except for the town judge) trusts the town judge.
* There is exactly one person that satisfies properties 1 and 2.

You are given trust, an array of pairs trust[i] = [a, b] representing that the person labelled a trusts the person labelled b.

If the town judge exists and can be identified, return the label of the town judge.  Otherwise, return -1.


Example 1:

	Input: N = 2, trust = [[1,2]]
	Output: 2	
	
Example 2:

	Input: N = 3, trust = [[1,3],[2,3]]
	Output: 3

Example 3:

	Input: N = 3, trust = [[1,3],[2,3],[3,1]]
	Output: -1
	
Example 4:

	Input: N = 3, trust = [[1,2],[2,3]]
	Output: -1
	
Example 5:

	Input: N = 4, trust = [[1,3],[1,4],[2,3],[2,4],[4,3]]
	Output: 3

Note:

	1 <= N <= 1000
	0 <= trust.length <= 10^4
	trust[i].length == 2
	trust[i] are all different
	trust[i][0] != trust[i][1]
	1 <= trust[i][0], trust[i][1] <= N

### 解析

根据题意，要找出隐藏在城镇中的法官，法官必须满足其他人都相信他，他不相信任何人。现在给出了 trust 列表表示 a 相信 b ，这样就可以对每个人进行计数。用长度为 N+1 的列表 d 表示所有人（索引为 0 的元素无用，其他索引代表城镇的每个人），然后遍历 trust 中的每个子列表 [a,b] ，d[a]-=1 表示他如果去相信其他人表示他肯定不是法官， d[b]+=1 表示他被人相信则可能是法官，遍历结束之后，只要 d 中的位置的计数为 N-1 ，就表示其为法官，将其索引返回，否则返回 -1 。

### 解答
				

	 class Solution(object):
	    def findJudge(self, N, trust):
	        """
	        :type N: int
	        :type trust: List[List[int]]
	        :rtype: int
	        """
	        d = [0] * (N+1)
	        for a,b in trust:
	            d[a] -= 1
	            d[b] += 1
	        for person in range(1, N + 1):
	            if d[person] == N - 1:
	                return person
	        return -1           	      
			
### 运行结果

	Runtime: 596 ms, faster than 96.12% of Python online submissions for Find the Town Judge.
	Memory Usage: 18.8 MB, less than 22.99% of Python online submissions for Find the Town Judge.


### 解析

另外在换一种思路，那就是我们维持两个长度为 N 列表 L 和 R （索引为 0 的元素无效），L[i] 表示第 i 个人得到的票数， R[i] 表示第 i 个人投出去多少票，法官肯定是一个得到 N-1 票，但是投出去 0 票的人，所以如果 N-1 在 L 中，那么找出其索引 idx ，然后再判断其对应的 R[idx] 是否为 0 ，如果是 0 那么表示 i 就是法官，否则直接返回 -1 即可。

### 解答

	class Solution(object):
	    def findJudge(self, N, trust):
	        """
	        :type N: int
	        :type trust: List[List[int]]
	        :rtype: int
	        """
	        L = [0] * (N+1)
	        for a, b in trust:
	            L[b] += 1
	        R = [0] * (N+1)
	        for a, b in trust:
	            R[a] += 1
	        if N-1 in L:
	            idx = L.index(N - 1, 1)
	            if R[idx] == 0:
	                return idx
	        return -1

### 运行结果

	Runtime: 751 ms, faster than 21.26% of Python online submissions for Find the Town Judge.
	Memory Usage: 18.8 MB, less than 29.90% of Python online submissions for Find the Town Judge.

原题链接：https://leetcode.com/problems/find-the-town-judge/


您的支持是我最大的动力
