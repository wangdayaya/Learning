leetcode 1043. Partition Array for Maximum Sum （python）

### 描述


Given an integer array arr, partition the array into (contiguous) subarrays of length at most k. After partitioning, each subarray has their values changed to become the maximum value of that subarray.

Return the largest sum of the given array after partitioning. Test cases are generated so that the answer fits in a 32-bit integer.




Example 1:

	Input: arr = [1,15,7,9,2,5,10], k = 3
	Output: 84
	Explanation: arr becomes [15,15,15,9,10,10,10]

	
Example 2:

	Input: arr = [1,4,1,5,7,3,6,1,9,9,3], k = 4
	Output: 83


Example 3:


	Input: arr = [1], k = 1
	Output: 1



Note:

	1 <= arr.length <= 500
	0 <= arr[i] <= 10^9
	1 <= k <= arr.length


### 解析


根据题意，给定一个整数数组 arr，将该数组划分为几个长度至多为 k 的连续子数组。 分区后，每个子数组的值都变为该子数组的最大值。题目要求我们分区后返回新数组可能的最大和。 题意很清楚，结合例子一基本也就能完全理解题意了。

* 使用动态规划，初始化长度为 len(arr) 的一维数组， dp[i]  表示的就是 arr[:i] 经过操作后得到的最大和值，
* 遍历 range(N) ，也就是遍历 arr 中的所有元素位置索引 i ，再遍历 range(i,max(-1, i-k),-1) 也就是可能的子数组位置索引 j ，不断更新当前子数组的最大值 MAX，又不断更新 dp[i] = max(dp[i], dp[j-1] + MAX*(i-j+1)) ，注意边界条件的限制。
* 遍历结束返回 dp[-1] 即为答案

### 解答
				
	class Solution(object):
	    def maxSumAfterPartitioning(self, arr, k):
	        """
	        :type arr: List[int]
	        :type k: int
	        :rtype: int
	        """
	        N = len(arr)
	        dp = [0 for _ in range(N)] 
	        for i in range(N):
	            MAX = 0
	            for j in range(i,max(-1, i-k),-1):
	                MAX = max(arr[j], MAX)
	                if j>=1:
	                    dp[i] = max(dp[i], dp[j-1] + MAX*(i-j+1))
	                else:
	                    dp[i] = max(dp[i], MAX*(i-j+1))
	        return dp[-1]
	                 
### 运行结果

	Runtime: 408 ms, faster than 61.04% of Python online submissions for Partition Array for Maximum Sum.
	Memory Usage: 13.4 MB, less than 94.81% of Python online submissions for Partition Array for Maximum Sum.



原题链接：https://leetcode.com/problems/partition-array-for-maximum-sum/

大神解释：https://www.bilibili.com/video/BV143411z7ek


您的支持是我最大的动力
