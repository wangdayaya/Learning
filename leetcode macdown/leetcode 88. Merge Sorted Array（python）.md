leetcode 88. Merge Sorted Array （python）



### 描述


You are given two integer arrays nums1 and nums2, sorted in non-decreasing order, and two integers m and n, representing the number of elements in nums1 and nums2 respectively.

Merge nums1 and nums2 into a single array sorted in non-decreasing order.

The final sorted array should not be returned by the function, but instead be stored inside the array nums1. To accommodate this, nums1 has a length of m + n, where the first m elements denote the elements that should be merged, and the last n elements are set to 0 and should be ignored. nums2 has a length of n.




Example 1:

	Input: nums1 = [1,2,3,0,0,0], m = 3, nums2 = [2,5,6], n = 3
	Output: [1,2,2,3,5,6]
	Explanation: The arrays we are merging are [1,2,3] and [2,5,6].
	The result of the merge is [1,2,2,3,5,6] with the underlined elements coming from nums1.

	
Example 2:

	Input: nums1 = [1], m = 1, nums2 = [], n = 0
	Output: [1]
	Explanation: The arrays we are merging are [1] and [].
	The result of the merge is [1].


Example 3:

	Input: nums1 = [0], m = 0, nums2 = [1], n = 1
	Output: [1]
	Explanation: The arrays we are merging are [] and [1].
	The result of the merge is [1].
	Note that because m = 0, there are no elements in nums1. The 0 is only there to ensure the merge result can fit in nums1.

	



Note:

	nums1.length == m + n
	nums2.length == n
	0 <= m, n <= 200
	1 <= m + n <= 200
	-10^9 <= nums1[i], nums2[j] <= 10^9


### 解析


根据题意，给定两个整数数组 nums1 和 nums2 ，按非递减顺序排序，以及两个整数 m 和 n ，分别表示 nums1 和 nums2 中的元素个数。将 nums1 和 nums2 合并为一个按非降序排序的数组。但是题目要求不用返回任何结果，而是直接在 nums1 上进行修改即可。

这道题如果非要强干，直接将两个数组合并成一个数组，然后直接调用内置函数 sort 进行递增排序即可，时间复杂度 O((m+n)log(m+n)) 和空间复杂度 O(m+n) ，肯定是符合题目要求的，但是这不满足题目要求的使用时间复杂度为 O(m+n) 。所以我们还是要换个思路。

这里我们知道两个数组都已经是有序数组，所以我们可以使用双指针，但是我们在往 nums1 中放置元素的时候会把已有的元素盖住，所以为了避免这种情况，我们使用双指针按照从后往前的顺序，不断进行比较，按照降序的顺序将元素都放置在 nums1 中。

时间复杂度为 O(m+n)，空间复杂度为 O(1) 。

### 解答
				
	class Solution(object):
	    def merge(self, nums1, m, nums2, n):
	        """
	        :type nums1: List[int]
	        :type m: int
	        :type nums2: List[int]
	        :type n: int
	        :rtype: None Do not return anything, modify nums1 in-place instead.
	        """
	        if n == 0: 
	            return nums1
	        if m == 0: 
	            nums1[:n] = nums2[:n]
	            return nums1
	        while m>0 and n>0:
	            if nums1[m-1]>=nums2[n-1]:
	                nums1[m+n-1] = nums1[m-1]
	                m -= 1
	            else:
	                nums1[m+n-1] = nums2[n-1]
	                n -= 1
	        nums1[:n] = nums2[:n]
	        return nums1
	                
	            

            	      
			
### 运行结果


	Runtime: 47 ms, faster than 12.32% of Python online submissions for Merge Sorted Array.
	Memory Usage: 13.6 MB, less than 5.52% of Python online submissions for Merge Sorted Array.

### 原题链接

	https://leetcode.com/problems/merge-sorted-array/

您的支持是我最大的动力
