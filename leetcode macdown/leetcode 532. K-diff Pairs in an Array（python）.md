leetcode  532. K-diff Pairs in an Array（python）




### 描述

Given an array of integers nums and an integer k, return the number of unique k-diff pairs in the array.

A k-diff pair is an integer pair (nums[i], nums[j]), where the following are true:

* 0 <= i < j < nums.length
* |nums[i] - nums[j]| == k

Notice that |val| denotes the absolute value of val.



Example 1:

	Input: nums = [3,1,4,1,5], k = 2
	Output: 2
	Explanation: There are two 2-diff pairs in the array, (1, 3) and (3, 5).
	Although we have two 1s in the input, we should only return the number of unique pairs.

	
Example 2:

	Input: nums = [1,2,3,4,5], k = 1
	Output: 4
	Explanation: There are four 1-diff pairs in the array, (1, 2), (2, 3), (3, 4) and (4, 5).


Example 3:

	Input: nums = [1,3,1,5,4], k = 0
	Output: 1
	Explanation: There is one 0-diff pair in the array, (1, 1).




Note:


	1 <= nums.length <= 10^4
	-10^7 <= nums[i] <= 10^7
	0 <= k <= 10^7

### 解析


根据题意，给定一个整数数组 nums 和一个整数 k，返回数组中唯一 k-diff 对的数量。 k-diff 对是一个整数对 (nums[i], nums[j])，其中满足以下条件：

* 0 <= i < j < nums.length
* |nums[i] - nums[j]| == k

这道题的题意看起来是很简单明了的，但是通过率只有 39% ，那是因为有一些边界的情况没有被发现，我做的时候也是报错了两次。首先我们要知道找到的整数对不能是重复的，所以计数一次情况就可以了，另外我们看题目中的限制条件就会发现 k 的值是可能为 0 的，如果 k>0 的情况，我们比较好处理，但是当 k==0 的时候我们要找的是同一个数字需要至少在 nums 中出现两次，而且要保证只计数一次。所以我们可以这样处理：

* 将 nums 升序排序，这样就能保证后面的数字肯定比前面的数字大，可以将第二个条件变为  nums[j] = nums[i] + k
* 然后我们定义一个集合 result 用来存放整数对，保证不重复
* 当 nums 不为空的时候执行 while 循环，然后每次循环 nums.pop(0) 得到最前面的元素 a ，当 a+k 在剩下的 nums 中，我们就将 (a, a+k) 加入到 result 中，一直到循环结束
* 最后只需要返回 result 的长度即可

时间复杂度为 O(N) ，空间复杂度为 O(N) 。其实我们可以看出来这种解法直接对 nums 进行排序，其实对于第一个条件 0 <= i < j < nums.length 没什么影响，因为第二个条件是个绝对值，元素的顺序是不影响的。

### 解答
				

	class Solution(object):
	    def findPairs(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        nums.sort()
	        result = set()
	        while nums:
	            a = nums.pop(0)
	            if a+k in nums:
	                result.add((a, a+k))
	        return len(result)
	

            	      
			
### 运行结果


	Runtime: 360 ms, faster than 14.77% of Python online submissions for K-diff Pairs in an Array.
	Memory Usage: 15.5 MB, less than 41.35% of Python online submissions for K-diff Pairs in an Array.

### 原题链接

https://leetcode.com/problems/k-diff-pairs-in-an-array/


您的支持是我最大的动力
