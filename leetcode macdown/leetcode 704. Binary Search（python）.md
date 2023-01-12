leetcode  704. Binary Search（python）




### 描述


Given an array of integers nums which is sorted in ascending order, and an integer target, write a function to search target in nums. If target exists, then return its index. Otherwise, return -1.

You must write an algorithm with O(log n) runtime complexity.


Example 1:

	Input: nums = [-1,0,3,5,9,12], target = 9
	Output: 4
	Explanation: 9 exists in nums and its index is 4

	
Example 2:


	Input: nums = [-1,0,3,5,9,12], target = 2
	Output: -1
	Explanation: 2 does not exist in nums so return -1



Note:

	1 <= nums.length <= 10^4
	-10^4 < nums[i], target < 10^4
	All the integers in nums are unique.
	nums is sorted in ascending order.


### 解析


根据题意，给定一个按升序排序的整数数组 nums 和一个整数目标，编写一个函数在 nums 中搜索目标。 如果目标存在，则返回其索引。 否则返回 -1 。题目已经明确规定让我们必须编写一个具有 O(log n) 运行时复杂度的算法。

一看题目的关键词 “数组”、“排序”、“搜索”、“目标”、“ O(log n)  时间复杂度”，那这道题考查的肯定就是二分查找法，正如题目所说的 “Binary Search” 。那么解决方法已经很明确了，就是使用二分查找法进行解题。思路很简单：

* 我们初始化两个指针 i 和 j 分别指向 nums 的头和尾
* 进行 while 循环，我们找到中间位置 mid = (i + j) // 2 ，然后进行比较，如果 nums[mid] 等于 target 那么就找到了目标直接返回索引即可；如果 nums[mid] 大于 target ，那说明 target 可能在前半段，所以要将 mid-1 赋与 j ；如果 nums[mid] 小于 target ，那说明 target 可能在后半段，所以要将 mid + 1 赋与 i ；
* 一直循环到当 i>j 的时候说明无法找到，直接返回 -1 即可

时间复杂度为 O(log n) ，空间复杂度为 O(1) 。

### 解答
				
	class Solution(object):
	    def search(self, nums, target):
	        i = 0
	        j = len(nums) - 1
	        while i <= j:
	            mid = (i + j) // 2
	            if nums[mid] == target:
	                return mid
	            elif nums[mid] > target:
	                j = mid - 1
	            else:
	                i = mid + 1
	        return -1

            	      
			
### 运行结果

	Runtime: 279 ms, faster than 41.48% of Python online submissions for Binary Search.
	Memory Usage: 14.6 MB, less than 78.69% of Python online submissions for Binary Search.


### 原题链接

https://leetcode.com/problems/binary-search/


您的支持是我最大的动力
