leetcode  2164. Sort Even and Odd Indices Independently（python）


### 前言

这是 Weekly Contest 279 比赛的第一题，难度 Easy ，考察的是对数组的操作，很简单。

### 描述


You are given a 0-indexed integer array nums. Rearrange the values of nums according to the following rules:

1. Sort the values at odd indices of nums in non-increasing order.

	* For example, if nums = [4,1,2,3] before this step, it becomes [4,3,2,1] after. The values at odd indices 1 and 3 are sorted in non-increasing order.

2. Sort the values at even indices of nums in non-decreasing order.
	* For example, if nums = [4,1,2,3] before this step, it becomes [2,1,4,3] after. The values at even indices 0 and 2 are sorted in non-decreasing order.
Return the array formed after rearranging the values of nums.


Example 1:

	Input: nums = [4,1,2,3]
	Output: [2,3,4,1]
	Explanation: 
	First, we sort the values present at odd indices (1 and 3) in non-increasing order.
	So, nums changes from [4,1,2,3] to [4,3,2,1].
	Next, we sort the values present at even indices (0 and 2) in non-decreasing order.
	So, nums changes from [4,1,2,3] to [2,3,4,1].
	Thus, the array formed after rearranging the values is [2,3,4,1].

	


Note:

	1 <= nums.length <= 100
	1 <= nums[i] <= 100


### 解析


根据题意，给你一个 0 索引的整数数组 nums。 根据以下规则重新排列 nums 的值：

1. 以降序顺序对 nums 奇数索引处的值进行排序
2. 以升序顺序对 nums 的偶数索引处的值进行排序

返回重新排列 nums 的值后形成的数组。

其实题意很简单，对于这种 Eazy 难度的题，我们只要顺着题意写代码就可以了。我们将索引为奇数的元素都放到 odd 列表中，将索引为偶数的元素都放到 even 列表中，然后对 odd 进行逆序排列，对 even 进行升序排序。然后初始化一个 result 列表，每次去弹出一个 even 最左边的元素，然后弹出一个 odd 最左边的元素，直到 even 和 odd 都为空，返回 result 即可。

时间复杂度为 O(logN) ，空间复杂度为 O(N) 。

其实像这种简单的题目，用 python 的话只需要一行代码就可以搞定，都写出来只不过是为了思路清晰。

### 解答
				

	class Solution(object):
	    def sortEvenOdd(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        odd = nums[1::2]
	        even = nums[::2]
	        odd.sort(reverse=True)
	        even.sort()
	        result = []
	        while even or odd:
	            if even:
	                result.append(even.pop(0))
	            if odd:
	                result.append(odd.pop(0))
	        return result
            	      
			
### 运行结果



	218 / 218 test cases passed.
	Status: Accepted
	Runtime: 71 ms
	Memory Usage: 13.4 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-279/problems/sort-even-and-odd-indices-independently/


您的支持是我最大的动力
