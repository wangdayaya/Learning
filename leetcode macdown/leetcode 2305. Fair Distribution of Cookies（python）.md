leetcode  2305. Fair Distribution of Cookies（python）




### 描述

You are given an integer array cookies, where cookies[i] denotes the number of cookies in the ith bag. You are also given an integer k that denotes the number of children to distribute all the bags of cookies to. All the cookies in the same bag must go to the same child and cannot be split up.

The unfairness of a distribution is defined as the maximum total cookies obtained by a single child in the distribution.

Return the minimum unfairness of all distributions.



Example 1:

	Input: cookies = [8,15,10,20,8], k = 2
	Output: 31
	Explanation: One optimal distribution is [8,15,8] and [10,20]
	- The 1st child receives [8,15,8] which has a total of 8 + 15 + 8 = 31 cookies.
	- The 2nd child receives [10,20] which has a total of 10 + 20 = 30 cookies.
	The unfairness of the distribution is max(31,30) = 31.
	It can be shown that there is no distribution with an unfairness less than 31.

	
Example 2:

	Input: cookies = [6,1,3,2,2,4,1,2], k = 3
	Output: 7
	Explanation: One optimal distribution is [6,1], [3,2,2], and [4,1,2]
	- The 1st child receives [6,1] which has a total of 6 + 1 = 7 cookies.
	- The 2nd child receives [3,2,2] which has a total of 3 + 2 + 2 = 7 cookies.
	- The 3rd child receives [4,1,2] which has a total of 4 + 1 + 2 = 7 cookies.
	The unfairness of the distribution is max(7,7,7) = 7.
	It can be shown that there is no distribution with an unfairness less than 7.





Note:

	2 <= cookies.length <= 8
	1 <= cookies[i] <= 10^5
	2 <= k <= cookies.length


### 解析

根据题意，给定个整数数组 cookies，其中 cookies[i] 表示第 i 个包中的 cookies 数量。 另外有一个整数 k ，表示将所有 cookie 袋分发给的孩子的数量。 同一个袋子里的所有饼干都必须给同一个孩子，不能分开。分配的不公平性定义为分配过程中单个孩子获得的 cookie 最大总和 。返回所有分配方式中的最小不公平性。

这道题是很典型的回溯题，我们如果使用暴力，找出 k^n 个不同的分法，这肯定是会超时的，我们可以在这个基础上进行优化，因为回溯本质上就是对暴力的优化，加速了暴力的运行。

定义一个长度为 k 的列表 L ，表示 k 个孩子的到的饼干数量，我们使用递归的方法，将每个饼干 cookies[i] 尝试给每个孩子 L[j] ，并且不断更新本次分配过程中过的最大值 val ，当将 cookies 分完的时候，我们取 val 更新 result ，再进行之后的递归操作，计算结束我们得到的 result 就是最后的答案。

时间复杂度为 O(k^n) ，空间复杂度为 O(k+n) ，这里的 k 是 L 的长度，n 是递归的深度 。


### 解答
				

	class Solution(object):
	    def __init__(self):
	        self.result = float('inf')
	
	    def distributeCookies(self, cookies, k):
	        N = len(cookies)
	        L = [0] * k
	
	        def dfs(i, val):
	            if val >= self.result:
	                return
	            if i == N:
	                self.result = min(self.result, val)
	                return
	            for j in range(k):
	                L[j] += cookies[i]
	                dfs(i + 1, max(val, L[j]))
	                L[j] -= cookies[i]
	
	        dfs(0, 0)
	        return self.result
            	      
			
### 运行结果

	Runtime: 45 ms, faster than 84.68% of Python online submissions for Fair Distribution of Cookies.
	Memory Usage: 13.5 MB, less than 44.35% of Python online submissions for Fair Distribution of Cookies.


### 原题链接


https://leetcode.com/contest/weekly-contest-297/problems/fair-distribution-of-cookies/


您的支持是我最大的动力
