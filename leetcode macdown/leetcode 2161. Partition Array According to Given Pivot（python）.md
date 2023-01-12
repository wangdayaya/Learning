leetcode  2161. Partition Array According to Given Pivot（python）


### 前言

这是 Biweekly Contest 71 比赛的第二题，难度 Medium  ，考察的是对题目的数组的重排序，也不难。

### 描述

You are given a 0-indexed integer array nums and an integer pivot. Rearrange nums such that the following conditions are satisfied:

* Every element less than pivot appears before every element greater than pivot.
* Every element equal to pivot appears in between the elements less than and greater than pivot.
* The relative order of the elements less than pivot and the elements greater than pivot is maintained.
* More formally, consider every pi, pj where pi is the new position of the ith element and pj is the new position of the jth element. For elements less than pivot, if i < j and nums[i] < pivot and nums[j] < pivot, then pi < pj. Similarly for elements greater than pivot, if i < j and nums[i] > pivot and nums[j] > pivot, then pi < pj.

Return nums after the rearrangement.



Example 1:

	Input: nums = [9,12,5,10,14,3,10], pivot = 10
	Output: [9,5,3,10,10,12,14]
	Explanation: 
	The elements 9, 5, and 3 are less than the pivot so they are on the left side of the array.
	The elements 12 and 14 are greater than the pivot so they are on the right side of the array.
	The relative ordering of the elements less than and greater than pivot is also maintained. [9, 5, 3] and [12, 14] are the respective orderings.

	




Note:

	1 <= nums.length <= 10^5
	-10^6 <= nums[i] <= 10^6
	pivot equals to an element of nums.


### 解析

根据题意，给出一个 0 索引的整数数组 nums 和一个整数 pivot 。 重新排列 nums 以满足以下条件：

* 每个小于 pivot 的元素出现在每个大于 pivot 的元素之前
* 每个等于 pivot 的元素都出现在小于和大于 pivot 的元素之间
* 保持小于 pivot 的元素和大于 pivot 的元素的相对顺序

说人话就是小于 pivot 的都在 pivot 的前面，大于 pivot 都在 pivot 的后面，等于 pivot 的元素在前面两部分的中间，并且小于 pivot 的部分的元素相对顺序不变，大于 pivot 的部分的元素的相对顺序不变，重新排列后返回 nums。

其实这个题目理解之后也很简单，就是初始化一个空列表 reuslt ：

* 第一次遍历 nums ，找出小于 pivot 的元素加入 result 中
* 第二次遍历 nums ，找出等于 pivot 的元素加入 result 中
* 第三次遍历 nums ，找出大于 pivot 的元素加入 result 中

最后将得到的 result 返回即可。时间复杂度为 O(N) ，空间复杂度为 O(N) 。然后耗时比较多但是也最后通过了，比赛的时候能用最快时间通过的就是好方法，哪顾得上去优化代码。
### 解答
				

	class Solution(object):
	    def pivotArray(self, nums, pivot):
	        """
	        :type nums: List[int]
	        :type pivot: int
	        :rtype: List[int]
	        """
	        result = []
	        for n in nums:
	            if n < pivot:
	                result.append(n)
	        for n in nums:
	            if n == pivot:
	                result.append(n)
	        for n in nums:
	            if n > pivot:
	                result.append((n))
	        return result
            	      
			
### 运行结果


	44 / 44 test cases passed.
	Status: Accepted
	Runtime: 1989 ms
	Memory Usage: 30.8 MB

### 原题链接


https://leetcode.com/contest/biweekly-contest-71/problems/partition-array-according-to-given-pivot/


您的支持是我最大的动力
