leetcode  2006. Count Number of Pairs With Absolute Difference K（python）

### 描述

Given an integer array nums and an integer k, return the number of pairs (i, j) where i < j such that |nums[i] - nums[j]| == k.

The value of |x| is defined as:

* x if x >= 0.
* -x if x < 0.




Example 1:


	Input: nums = [1,2,2,1], k = 1
	Output: 4
	Explanation: The pairs with an absolute difference of 1 are:
	- [1,2,2,1]
	- [1,2,2,1]
	- [1,2,2,1]
	- [1,2,2,1]
	
Example 2:

	Input: nums = [1,3], k = 3
	Output: 0
	Explanation: There are no pairs with an absolute difference of 3.


Example 3:

	
	Input: nums = [3,2,1,5,4], k = 2
	Output: 3
	Explanation: The pairs with an absolute difference of 2 are:
	- [3,2,1,5,4]
	- [3,2,1,5,4]
	- [3,2,1,5,4]
	




Note:

	1 <= nums.length <= 200
	1 <= nums[i] <= 100
	1 <= k <= 99


### 解析

根据题意，就是给出了一个整数列表 nums ，然后给出了一个整数 k ，要求让我们找索引对 ( i , j ) ，且 i < j ，同时满足 |nums[i] - nums[j]| == k ，问我们一共有多少对这样的索引。思路很简单，对于这么简单的问题，自然是先暴力大法先解决咯。

* 初始化结果 result 为 0 , nums 的长度为 M
* 第一层循环遍历 ( 0 , M-1)，对于每一个索引 i ，再进行第二层循环遍历 ( i+1 ，M) 
* 当 nums[i]-nums[j] == k 或者 nums[j]-nums[i] == k 的时候 result 加一
* 遍历结束得到的 result 为答案

### 解答
				
	class Solution(object):
	    def countKDifference(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        result = 0
	        M = len(nums)
	        for i in range(M-1):
	            for j in range(i+1,M):
	                if nums[i]-nums[j] == k or  nums[j]-nums[i] == k:
	                    result += 1
	        return result
	                    

            	      
			
### 运行结果


	Runtime: 264 ms, faster than 15.12% of Python online submissions for Count Number of Pairs With Absolute Difference K.
	Memory Usage: 13.4 MB, less than 67.23% of Python online submissions for Count Number of Pairs With Absolute Difference K.
	
	
### 解析

我们还可以用 python 的内置函数 itertools.combinations 来解决本题，我们需要使用内置函数组成长度 2 的各种组合，所以直接调用 itertools.combinations(nums, 2) 即可得到迭代器，遍历迭代器中每个组合，判断两个元素的绝对值差为 k 则结果 result 加一，遍历结束得到的 result 为答案。

函数举例：

> 运行 [(a,b) for a,b in itertools.combinations([1,2,2,1], 2)]
> 
> 答案 [(1, 2), (1, 2), (1, 1), (2, 2), (2, 1), (2, 1)]，可以看出来使用内置函数得到的列表中元素的组合，已经有了前后的顺序，这样我们就不用关心 i<j 的问题了


### 解答
				
	class Solution(object):
	    def countKDifference(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        result = 0
	        for a,b in itertools.combinations(nums, 2):
	            if a-b ==k or b-a==k:
	                result += 1
	        return result

            	      
			
### 运行结果

	Runtime: 196 ms, faster than 57.56% of Python online submissions for Count Number of Pairs With Absolute Difference K.
	Memory Usage: 13.7 MB, less than 5.04% of Python online submissions for Count Number of Pairs With Absolute Difference K.

原题链接：https://leetcode.com/problems/count-number-of-pairs-with-absolute-difference-k/



您的支持是我最大的动力
