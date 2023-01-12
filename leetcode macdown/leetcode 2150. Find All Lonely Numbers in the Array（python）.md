leetcode  2150. Find All Lonely Numbers in the Array（python）


### 前言



其实这次的单周赛前三道题都很简单，虽然第二题和第三题标注的难度是 Medium ，但是其实感觉有点夸大了，最多就是个 Eazy 难度。这是 Weekly Contest 277 的第三题，难度 Medium ，考察的是对列表的排序和一些基本操作，没啥好说的。

### 描述


You are given an integer array nums. A number x is lonely when it appears only once, and no adjacent numbers (i.e. x + 1 and x - 1) appear in the array.

Return all lonely numbers in nums. You may return the answer in any order.


Example 1:


	Input: nums = [10,6,5,8]
	Output: [10,8]
	Explanation: 
	- 10 is a lonely number since it appears exactly once and 9 and 11 does not appear in nums.
	- 8 is a lonely number since it appears exactly once and 7 and 9 does not appear in nums.
	- 5 is not a lonely number since 6 appears in nums and vice versa.
	Hence, the lonely numbers in nums are [10, 8].
	Note that [8, 10] may also be returned.



Note:


	1 <= nums.length <= 10^5
	0 <= nums[i] <= 10^6

### 解析

根据题意，给你一个整数数组 nums。 一个数字 x 只出现一次时，并且数组中没有相邻的数字（即 x + 1 和 x - 1）出现，那么这个数字就是孤独的。返回 nums 中的所有孤独数字。可以按任何顺序返回答案。

这道题基本上读完之后基本就有思路了，要找某个元素是不是孤独的就是满足两个条件：

* 只出现过一次
* 并且没有相邻的数字出现在 nums 中

所以我们最简单的方法就是对 nums 进行升序排序，这样对于某个元素，小于等于其的元素肯定在左边，大于等于其的元素肯定在右边，所以我们只要遍历每个元素，只要某个元素满足 nums[i]-nums[i-1]>1 and nums[i+1]-nums[i]>1 就能将其加入到结果列表 result 中。但是这里还是有三个需要注意的点：

* 如果列表的长度为 1 ，那么他肯定是孤独的，直接返回 nums 即可
* 在经过对 nums 排序之后，对于第一个元素和最后一个，我们要单独判断其是否是孤独的，因为对于第一个元素是没有小于等于其的元素存在，对于最后一个元素是没有大于等于其的元素存在
* 另外我们的代码中也没有去判断某个元素的出现次数是否为 1 ，因为在经过对 nums 排序之后，nums[i]-nums[i-1]>1 and nums[i+1]-nums[i]>1 这个条件就可以保证这三个相邻的元素是单调递增的，而该元素肯定是只出现过一次，这是一个很关键的技巧，否则去对每个元素计数又要多耗时和耗空间了

这个算法的时间复杂度是 O(N) ，空间复杂度为 O(N)。


### 解答
				

	class Solution(object):
	    def findLonely(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        N = len(nums)
	        if N==1: return nums
	        nums.sort()
	        result = []
	        if nums[1]-nums[0]>1:
	            result.append(nums[0])
	        if nums[-1]-nums[-2]>1:
	            result.append(nums[-1])
	        for i in range(1, N-1):
	            if nums[i]-nums[i-1]>1 and nums[i+1]-nums[i]>1:
	                result.append(nums[i])
	        return result
	        
            	      
			
### 运行结果

	75 / 75 test cases passed.
	Status: Accepted
	Runtime: 2007 ms
	Memory Usage: 31.2 MB


### 原题链接


https://leetcode.com/contest/weekly-contest-277/problems/find-all-lonely-numbers-in-the-array/


您的支持是我最大的动力
