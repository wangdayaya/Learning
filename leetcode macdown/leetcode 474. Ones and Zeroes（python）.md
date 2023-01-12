leetcode 474. Ones and Zeroes （python）




### 描述


You are given an array of binary strings strs and two integers m and n.

Return the size of the largest subset of strs such that there are at most m 0's and n 1's in the subset.

A set x is a subset of a set y if all elements of x are also elements of y.


Example 1:

	Input: strs = ["10","0001","111001","1","0"], m = 5, n = 3
	Output: 4
	Explanation: The largest subset with at most 5 0's and 3 1's is {"10", "0001", "1", "0"}, so the answer is 4.
	Other valid but smaller subsets include {"0001", "1"} and {"10", "1", "0"}.
	{"111001"} is an invalid subset because it contains 4 1's, greater than the maximum of 3.

	
Example 2:


	Input: strs = ["10","0","1"], m = 1, n = 1
	Output: 2
	Explanation: The largest subset is {"0", "1"}, so the answer is 2.






Note:

	1 <= strs.length <= 600
	1 <= strs[i].length <= 100
	strs[i] consists only of digits '0' and '1'.
	1 <= m, n <= 100


### 解析

根据题意，给定一个二进制字符串数组 strs 和两个整数 m 和 n 。返回 strs 的最大子集的大小，使得子集中最多有 m 个 0 和 n 个 1 。 如果 x 的所有元素也是 y 的元素，则集合 x 是集合 y 的子集。


这道题目本质上就是在考察类似背包问题的动态规划，只不过这里的限制条件成了 0 和 1 的个数，我们就按照题目的问题，设置动态规划 dp[i][j][k] 表示在前 i 个元素中选择最多数量的字符串，使得 0 的数目不超过 j ，1 的数量不超过 k 。状态转移方程为：

	dp[i][j][k] = max(dp[i-1][j][k], dp[i-1][j-z][k-o]+1) ，z 表示 0 的数量， o 表示 1 的数量
	
时间复杂度为 O(N\*m\*n)，空间复杂度为 O(N\*m\*n) 。	


### 解答
				
	class Solution(object):
	    def findMaxForm(self, strs, m, n):
	        """
	        :type strs: List[str]
	        :type m: int
	        :type n: int
	        :rtype: int
	        """
	        N = len(strs)
	        dp = [[[0 for _ in range(n + 1)] for _ in range(m + 1)] for _ in range(N + 1)]
	        for i in range(1, N + 1):
	            count_0 = strs[i - 1].count('0')
	            count_1 = strs[i - 1].count('1')
	            for j in range(m + 1):
	                for k in range(n + 1):
	                    if j < count_0 or k < count_1:
	                        dp[i][j][k] = dp[i - 1][j][k]
	                    else:
	                        dp[i][j][k] = max(dp[i - 1][j][k], dp[i - 1][j - count_0][k - count_1] + 1)
	        return dp[N][m][n]


        

            	      
			
### 运行结果


	Runtime: 5700 ms, faster than 13.79% of Python online submissions for Ones and Zeroes.
	Memory Usage: 60.5 MB, less than 32.18% of Python online submissions for Ones and Zeroes.

### 原题链接



https://leetcode.com/problems/ones-and-zeroes/


您的支持是我最大的动力
