leetcode 2248. Intersection of Multiple Arrays（python）

这第 290 场 leetcode 周赛，本文介绍的是第一题，难度 Eazy ，主要考察的是对集合的相关操作。


### 描述

Given a 2D integer array nums where nums[i] is a non-empty array of distinct positive integers, return the list of integers that are present in each array of nums sorted in ascending order.



Example 1:

	Input: nums = [[3,1,2,4,5],[1,2,3,4],[3,4,5,6]]
	Output: [3,4]
	Explanation: 
	The only integers present in each of nums[0] = [3,1,2,4,5], nums[1] = [1,2,3,4], and nums[2] = [3,4,5,6] are 3 and 4, so we return [3,4].

	
Example 2:

	Input: nums = [[1,2,3],[4,5,6]]
	Output: []
	Explanation: 
	There does not exist any integer present both in nums[0] and nums[1], so we return an empty list [].






Note:

	1 <= nums.length <= 1000
	1 <= sum(nums[i].length) <= 1000
	1 <= nums[i][j] <= 1000
	All the values of nums[i] are unique.


### 解析

根据题意，给定一个二维整数数组 nums，其中 nums[i] 是不同正整数的非空数组，题目要求我们找出在每个 nums[i] 中都出现的元素列表，并将他们按照升序的顺序返回。

这道题很简单，我用了十分钟才做出来，因为看英文的题目太快，自己想太多，把题目想复杂了，所以折腾了很久，所以理解清楚题意是极其重要的，否则就是浪费时间。

解决思路也比较简单，既然要找出存在于所有 nums[i] 中的元素，换句话说就是对所有的 nums[i] 进行求交集的操作，得到的结果进行升序排序即可，这里有一个小技巧，因为 nums[i][j] 的最大值为 1000 ，所以我初始化一个集合 result 里面包含了从 1 到 1000 的所有整数，只需要去和每个 nums[i] 进行求交集更新 result 即可，最后将得到的 result 进行升序排列返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				

	class Solution(object):
	    def intersection(self, nums):
	        """
	        :type nums: List[List[int]]
	        :rtype: List[int]
	        """
	        result = set(i for i in range(1001))
	        for num in nums:
	            result &= set(num)
	        return sorted(list(result))
	        
            	      
			
### 运行结果


	151 / 151 test cases passed.
	Status: Accepted
	Runtime: 96 ms
	Memory Usage: 13.8 MB


### 原题链接


https://leetcode.com/contest/weekly-contest-290/problems/intersection-of-multiple-arrays/


您的支持是我最大的动力
