leetcode  2344. Minimum Deletions to Make Array Divisible（python）




### 描述

You are given two positive integer arrays nums and numsDivide. You can delete any number of elements from nums. Return the minimum number of deletions such that the smallest element in nums divides all the elements of numsDivide. If this is not possible, return -1. Note that an integer x divides y if y % x == 0.



Example 1:


	Input: nums = [2,3,2,4,3], numsDivide = [9,6,9,3,15]
	Output: 2
	Explanation: 
	The smallest element in [2,3,2,4,3] is 2, which does not divide all the elements of numsDivide.
	We use 2 deletions to delete the elements in nums that are equal to 2 which makes nums = [3,4,3].
	The smallest element in [3,4,3] is 3, which divides all the elements of numsDivide.
	It can be shown that 2 is the minimum number of deletions needed.
	
Example 2:


	Input: nums = [4,3,6], numsDivide = [8,2,6,10]
	Output: -1
	Explanation: 
	We want the smallest element in nums to divide all the elements of numsDivide.
	There is no way to delete elements from nums to allow this.




Note:

	1 <= nums.length, numsDivide.length <= 10^5
	1 <= nums[i], numsDivide[i] <= 10^9


### 解析

根据题意，给定两个正整数数组 nums 和 numsDivide。 可以从 nums 中删除任意数量的元素。 返回最小删除次数，使得 nums 中的最小元素可以除以 numsDivide 的所有元素。 如果这不可能，则返回 -1 。 

本题虽然是难度为 Hard ，但是其实只是个初级 Medium 的题目，我们可以先找出所有 numsDivide 元素的最大公约数 g ，然后我们对 nums 进行从小到大的排序，然后我们遍历 nums 中的元素 n 及其索引，因为要使一个数字能够除以所有的 numsDivide ，那么这个数字肯定能除以 g ，所以只要  g % n 为 0 ，那么只需要将当前的索引返回，也就是需要删除的元素个数。

时间复杂度为 O(MlogA+NlogN+N)，其中 M 为数组 numsDivide 的长度，A=max(numsDivide)，N 为数组 nums 的长度，空间复杂度为 O(N) 。


### 解答

	class Solution:
	    def minOperations(self, nums: List[int], numsDivide: List[int]) -> int:
	        def cal_gcd(L):
	            result = 1
	            L = list(set(L))
	            for i in range(len(L)):
	                if i == 0:
	                    result = L[0]
	                else:
	                    result = math.gcd(result, L[i])
	            return result
	
	        g = cal_gcd(numsDivide)
	        nums.sort()
	        for i, n in enumerate(nums):
	            if g % n == 0:
	                return i
	        return -1

### 运行结果

	39 / 39 test cases passed.
	Status: Accepted
	Runtime: 1650 ms
	Memory Usage: 31 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-302/problems/minimum-deletions-to-make-array-divisible/


您的支持是我最大的动力
