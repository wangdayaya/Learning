leetcode  2215. Find the Difference of Two Arrays（python）




### 描述

Given two 0-indexed integer arrays nums1 and nums2, return a list answer of size 2 where:

* answer[0] is a list of all distinct integers in nums1 which are not present in nums2.
* answer[1] is a list of all distinct integers in nums2 which are not present in nums1.

Note that the integers in the lists may be returned in any order.



Example 1:

	Input: nums1 = [1,2,3], nums2 = [2,4,6]
	Output: [[1,3],[4,6]]
	Explanation:
	For nums1, nums1[1] = 2 is present at index 0 of nums2, whereas nums1[0] = 1 and nums1[2] = 3 are not present in nums2. Therefore, answer[0] = [1,3].
	For nums2, nums2[0] = 2 is present at index 1 of nums1, whereas nums2[1] = 4 and nums2[2] = 6 are not present in nums2. Therefore, answer[1] = [4,6].

	



Note:

	1 <= nums1.length, nums2.length <= 1000
	-1000 <= nums1[i], nums2[i] <= 1000


### 解析

根据题意，给定两个索引为 0 的整数数组 nums1 和 nums2，返回大小为 2 的列表 answer ，列表中的整数可以按任何顺序返回，结果要满足两个条件：

* answer[0] 是 nums1 中不存在于 nums2 中的所有不同整数的列表
* answer[1] 是 nums2 中不存在于 nums1 中的所有不同整数的列表

这很明显就是对数组的简单遍历和判断，看题目的限制条件，我们发现 nums1 和 nums2 的长度最大是 1000 ，我们只需要暴力求解即可，先遍历 nums1 中的元素，找不存在于 nums2 中的元素去重之后都放入一个列表 a 中，然后遍历 nums2 中的元素，找不存在于 nums1 中的元素去重之后都放入一个列表 b 中，最后将 a 和 b 拼接成结果列表返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				

	class Solution(object):
	    def findDifference(self, nums1, nums2):
	        """
	        :type nums1: List[int]
	        :type nums2: List[int]
	        :rtype: List[List[int]]
	        """
	        a = []
	        for c in nums1:
	            if c not in nums2 and c not in a:
	                a.append(c)
	        b = []
	        for c in nums2:
	            if c not in nums1 and c not in b:
	                b.append(c)
	        return [a,b]
	                
            	      
			
### 运行结果

	202 / 202 test cases passed.
	Status: Accepted
	Runtime: 930 ms
	Memory Usage: 13.6 MB


### 解析
当然了上面的解法纯粹是为了无脑快速解题的暴力解法，其实这种题意阐述的相当明确，最简洁的方法肯定是用集合求差集。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

通过比较两种解法的耗时，我们发现这种解法更加节省时间。

### 解答

	class Solution(object):
	    def findDifference(self, nums1, nums2):
	        """
	        :type nums1: List[int]
	        :type nums2: List[int]
	        :rtype: List[List[int]]
	        """
	        s1 = set(nums1)
	        s2 = set(nums2)
	        a = list(s1-s2)
	        b = list(s2-s1)
	        return [a,b]
### 运行结果

	202 / 202 test cases passed.
	Status: Accepted
	Runtime: 193 ms
	Memory Usage: 13.7 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-286/problems/find-the-difference-of-two-arrays/


您的支持是我最大的动力
