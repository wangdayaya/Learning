leetcode 823. Binary Trees With Factors  （python）




### 描述

Given an array of unique integers, arr, where each integer arr[i] is strictly greater than 1.

We make a binary tree using these integers, and each number may be used for any number of times. Each non-leaf node's value should be equal to the product of the values of its children.

Return the number of binary trees we can make. The answer may be too large so return the answer modulo 10^9 + 7.





Example 1:

	Input: arr = [2,4]
	Output: 3
	Explanation: We can make these trees: [2], [4], [4, 2, 2]

	
Example 2:


	Input: arr = [2,4,5,10]
	Output: 7
	Explanation: We can make these trees: [2], [4], [5], [10], [4, 2, 2], [10, 2, 5], [10, 5, 2].



Note:


	1 <= arr.length <= 1000
	2 <= arr[i] <= 10^9
	All the values of arr are unique.

### 解析
根据题意，给定一个唯一整数数组 arr，其中每个整数 arr[i] 严格大于 1 。我们使用这些整数制作一棵二叉树，每个数字可以使用任意次数。 每个非叶节点的值应该等于其子节点值的乘积。返回我们可以制作的二叉树的数量。 答案可能太大，因此以 10^9 + 7 为模返回答案。

这道题很明显就是考察动态规划，根据题目的描述，我们用朴实的想法肯定会想到因为整数乘法的原因，树越往根结点的值会越大，所以我们可以定义 dp[v] 表示当整数 v 为根结点的时候的不同构造数量。由于每个节点有两个子节点 x 和 y ，x \* y = v ，所以我们可以找出转移方程 dp[v] = dp[x] \* dp[y] 。

我们将 arr 按照升序排序，然后对于每一个 arr[i] ，我们去遍历小于索引 i 的所有元素，如果找到有 x \* y = arr[i] ，我们更新 dp[i] 即可。

时间复杂度为 O(N^2) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def numFactoredBinaryTrees(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: int
	        """
	        arr.sort()
	        MOD = 10 ** 9 + 7
	        dp = [1] * len(arr)
	        idx = {c: i for i, c in enumerate(arr)}
	        for i, c in enumerate(arr):
	            for j in range(i):
	                if c % arr[j] == 0:
	                    right = c / arr[j]
	                    if right in idx:
	                        dp[i] += dp[j] * dp[idx[right]]
	                        dp[i] %= MOD
	        return sum(dp) % MOD

### 运行结果

	Runtime: 498 ms, faster than 69.23% of Python online submissions for Binary Trees With Factors.
	Memory Usage: 13.6 MB, less than 69.23% of Python online submissions for Binary Trees With Factors.


### 原题链接


https://leetcode.com/problems/binary-trees-with-factors/

您的支持是我最大的动力
