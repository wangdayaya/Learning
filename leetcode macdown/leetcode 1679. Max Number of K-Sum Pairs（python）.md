leetcode 1679. Max Number of K-Sum Pairs （python）




### 描述

You are given an integer array nums and an integer k.

In one operation, you can pick two numbers from the array whose sum equals k and remove them from the array.

Return the maximum number of operations you can perform on the array.

 



Example 1:

	Input: nums = [1,2,3,4], k = 5
	Output: 2
	Explanation: Starting with nums = [1,2,3,4]:
	- Remove numbers 1 and 4, then nums = [2,3]
	- Remove numbers 2 and 3, then nums = []
	There are no more pairs that sum up to 5, hence a total of 2 operations.

	
Example 2:

	Input: nums = [3,1,3,4,3], k = 6
	Output: 1
	Explanation: Starting with nums = [3,1,3,4,3]:
	- Remove the first two 3's, then nums = [1,4,3]
	There are no more pairs that sum up to 6, hence a total of 1 operation.



Note:

	1 <= nums.length <= 10^5
	1 <= nums[i] <= 10^9
	1 <= k <= 10^9


### 解析


根据题意，给定一个整数数组 nums 和一个整数 k 。在一次操作中，可以从数组中挑选两个总和等于 k 的数字并将它们从数组中删除。返回可以对数组执行的最大操作数。

这道题其实就是考察的是一个字典的实际应用，我们知道两个和为 k 的数字其实可以是一样的，也可以是不一样的，对于一样的数字来说，我们最多可以执行的次数无非就是 count[n] // 2 ；对于不一样的数字来说我们执行的次数就是两个数字中的出现较少次数的那一个 min(count[n], count[k-n]) ，前提是两个数字都在 nums 中有出现，到这里思路都已经讲完了。

我们在写代码的时候只需要先对 nums 中的元素进行计数，获得计数器 c ，然后遍历 c 中的 key-value 对，如果 k-key 和 key 相等，则直接将 value // 2 加入 result 中，如果不相等，则取较少出现次数 min(value, c[k - key]) 加入 result 中，这里有一点需要注意的是要避免重复计数，因为第二种情况下 key 和 k-key 都在 count 中，我们只需要加入判断条件 key*2<k 就能保证只进行一次计数，最后将结果 result 返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答
				

	class Solution(object):
	    def maxOperations(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        c = collections.Counter(nums)
	        result = 0
	        for key, value in c.items():
	            if key * 2 == k:
	                result += value // 2
	            elif key * 2 < k and k - key in c:
	                result += min(value, c[k - key])
	        return result

            	      
			
### 运行结果

	Runtime: 615 ms, faster than 85.21% of Python online submissions for Max Number of K-Sum Pairs.
	Memory Usage: 24.9 MB, less than 6.52% of Python online submissions for Max Number of K-Sum Pairs.




### 原题链接


https://leetcode.com/problems/max-number-of-k-sum-pairs/


您的支持是我最大的动力
