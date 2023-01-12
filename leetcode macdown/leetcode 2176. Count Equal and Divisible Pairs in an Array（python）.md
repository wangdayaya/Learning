

### 前言

这是 leetcode 中 Biweekly Contest 72 的第一题，难度为 Eazy ，考查的就是对列表的基本操作。其实这次的第 72 场双周赛难度都比较低，我在不到半个小时的时间里都做出来三道题，平时我可是全程下来最多也只能做三道题。


### 描述


Given a 0-indexed integer array nums of length n and an integer k, return the number of pairs (i, j) where 0 <= i < j < n, such that nums[i] == nums[j] and (i * j) is divisible by k.


Example 1:


Input: nums = [3,1,2,2,2,1,3], k = 2
Output: 4
Explanation:
There are 4 pairs that meet all the requirements:
- nums[0] == nums[6], and 0 * 6 == 0, which is divisible by 2.
- nums[2] == nums[3], and 2 * 3 == 6, which is divisible by 2.
- nums[2] == nums[4], and 2 * 4 == 8, which is divisible by 2.
- nums[3] == nums[4], and 3 * 4 == 12, which is divisible by 2.
	


Note:

* 1 <= nums.length <= 100
* 1 <= nums[i], k <= 100






### 解析


根据题意，给定一个长度为 n 的 0 索引整数数组 nums 和一个整数 k，返回  (i, j)  的个数，必须要满足 0 <= i < j < n ，且使得 nums[i] == nums[j] 和 ( i * j) 可被 k 整除。

这题意说的是很清楚了，条件给出的也很明晰，而且限制条件也指出来 nums 的最长的长度为 1000 ，nums 中的元素和 k 的值最大为 100 ，所以基本上是可以使用暴力直接求解的，算是比赛的送分题目吧，这个都做不出来真的完蛋了。

初始化一个结果列表 result ，然后双重 for 循环遍历所有的 nums 中的索引对 (i,j) ，如果满足： 

    nums[i] == nums[j] and (i*j)%k==0 and [i,j] not in result 
就将 (i,j) 加入 result ，遍历结束最后返回 result 的长度即可。

### 解答
				

	class Solution(object):
	    def countPairs(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        result = []
	        N = len(nums)
	        for i in range(N):
	            for j in range(i+1, N):
	                if nums[i] == nums[j] and (i*j)%k==0 and [i,j] not in result:
	                    result.append([i,j])
	        return len(result)
            	      
			
### 运行结果

	
	237 / 237 test cases passed.
	Status: Accepted
	Runtime: 858 ms
	Memory Usage: 13.9 MB


### 原题链接


https://leetcode.com/contest/biweekly-contest-72/problems/count-equal-and-divisible-pairs-in-an-array/


您的支持是我最大的动力
