leetcode  2190. Most Frequent Number Following Key In an Array（python）

### 前言

这是 Biweekly Contest 73 的第一题，难度 Eazy ，考察的就是对列表的操作和对计数器的使用。



### 描述



You are given a 0-indexed integer array nums. You are also given an integer key, which is present in nums.

For every unique integer target in nums, count the number of times target immediately follows an occurrence of key in nums. In other words, count the number of indices i such that:

* 0 <= i <= nums.length - 2,
* nums[i] == key and,
* nums[i + 1] == target.

Return the target with the maximum count. The test cases will be generated such that the target with maximum count is unique.

 

Example 1:

	Input: nums = [1,100,200,1,100], key = 1
	Output: 100
	Explanation: For target = 100, there are 2 occurrences at indices 1 and 4 which follow an occurrence of key.
	No other integers follow an occurrence of key, so we return 100.



Note:

	2 <= nums.length <= 1000
	1 <= nums[i] <= 1000
	The test cases will be generated such that the answer is unique.


### 解析

不得不吐槽这第一道题，不知是哪位天才出的，我 TM 当时比赛的时候硬是没看懂英文题目描述讲了什么，最后没办法结合例子试着做题，报错了两次才明白这道题到底是要说什么，真的是醉了。

其实很简单，就是在一个 0 索引的整数列表 nums 中，同时又给了一个肯定存在与 nums 中的整数 key ，然后让我们找在 nums 中所有的 key 后面紧挨的一个元素所能形成的整数集合中，出现次数最多的元素是哪个。我的天，现在解读这道题意仍然有点吃力。

明白了题意，解题的思路其实很简答，我们从左到右遍历 nums 中的每个元素 c （索引为 i ），如果等于 key ，并且当前元素的索引小于 len(nums)-1 ，那么我们就将 nums[i+1] 加入到结果列表 result 中，最后只需要通过函数 Counter 进行计数，找出出现最多的元素即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。






### 解答
				
	class Solution(object):
	    def mostFrequent(self, nums, key):
	        """
	        :type nums: List[int]
	        :type key: int
	        :rtype: int
	        """
	        result = []
	        for i, c in enumerate(nums):
	            if c == key and i+1<len(nums):
	                result.append(nums[i+1])
	        return collections.Counter(result).most_common()[0][0]

            	      
			
### 运行结果


	94 / 94 test cases passed.
	Status: Accepted
	Runtime: 114 ms
	Memory Usage: 13.8 MB

### 原题链接

https://leetcode.com/contest/biweekly-contest-73/problems/most-frequent-number-following-key-in-an-array/


您的支持是我最大的动力
