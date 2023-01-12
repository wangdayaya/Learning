leetcode  216. Combination Sum III（python）




### 描述


Find all valid combinations of k numbers that sum up to n such that the following conditions are true:

Only numbers 1 through 9 are used.
Each number is used at most once.
Return a list of all possible valid combinations. The list must not contain the same combination twice, and the combinations may be returned in any order.


Example 1:

	Input: k = 3, n = 7
	Output: [[1,2,4]]
	Explanation:
	1 + 2 + 4 = 7
	There are no other valid combinations.

	
Example 2:

	Input: k = 3, n = 9
	Output: [[1,2,6],[1,3,5],[2,3,4]]
	Explanation:
	1 + 2 + 6 = 9
	1 + 3 + 5 = 9
	2 + 3 + 4 = 9
	There are no other valid combinations.


Example 3:


	Input: k = 4, n = 1
	Output: []
	Explanation: There are no valid combinations.
	Using 4 different numbers in the range [1,9], the smallest sum we can get is 1+2+3+4 = 10 and since 10 > 1, there are no valid combination.
	



Note:

	2 <= k <= 9
	1 <= n <= 60


### 解析


根据题意，求所有和为 n 且满足下列条件的 k 个数的所有有效组合：

* 仅使用数字 1 到 9
* 每个号码最多使用一次

返回所有可能的有效组合的列表。 该列表不能包含两个相同的组合，并且可以以任何顺序返回组合。

这道题很明显考察的是 DFS ，我们在找数字组合的时候，首先要排除一些不可能的情况，也就是当最小的 k 个数字的和大于 n 的时候，说明根本没有可能的组合，直接返回空列表即可，当最小的 k 个数字的和等于 n 的时候，说明只能有一组数字符和要求，直接返回这个列表即可。其实用 DFS 也能做这些事情，但是直接提前写出来会提升 AC 的运行速度，减少运行时间，和直接用 DFS 相比时间会缩短 20 ms 。


剩下的情况就交给 DFS 去解决，k 相当于树的深度， n 相当于树的宽度，我们定义递归函数 dfs(i, k, n, L) ：

*  L 来存放一个可能组合中已经找到的数字列表
*  n 表示剩余列表的和
*  k 表示 L 中还差几个数字
*  i 表示我们下一层 for 循环开始搜索数字的起始位置
*  终止条件就是 当 n == 0 且 k == 0 的时候表示 L 中的组合是符合题意的，直接将其加入到 result 中即可

时间复杂度就是排列的数量 O(C(9, k))  ，总共可能有 9 次递归栈，每次里面的列表 L 所需空间最多为 k ，所以空间复杂度为 O(9 + k) 。


### 解答
				
	class Solution(object):
	    def combinationSum3(self, k, n):
	        """
	        :type k: int
	        :type n: int
	        :rtype: List[List[int]]
	        """
	        if sum(range(1, k + 1)) > n:
	            return []
	        if sum(range(1, k + 1)) == n:
	            return [[i for i in range(1, k + 1)]]
	        result = []
	        def dfs(i, k, n, L):
	            if n == 0 and k == 0:
	                result.append(L)
	                return
	            for i in range(i, min(10,n + 1)):
	                dfs(i + 1, k - 1, n - i, L + [i])
	        dfs(1, k, n, [])
	        return result
            	      
			
### 运行结果

	Runtime: 21 ms, faster than 71.51% of Python online submissions for Combination Sum III.
	Memory Usage: 13.2 MB, less than 89.83% of Python online submissions for Combination Sum III.


### 原题链接



https://leetcode.com/problems/combination-sum-iii/


您的支持是我最大的动力
