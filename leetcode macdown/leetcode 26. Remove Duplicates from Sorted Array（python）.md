leetcode  26. Remove Duplicates from Sorted Array（python）

### 描述

Given an integer array nums sorted in non-decreasing order, remove the duplicates in-place such that each unique element appears only once. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

**Custom Judge**:

The judge will test your solution with the following code:

	int[] nums = [...]; // Input array
	int[] expectedNums = [...]; // The expected answer with correct length
	
	int k = removeDuplicates(nums); // Calls your implementation
	
	assert k == expectedNums.length;
	for (int i = 0; i < k; i++) {
	    assert nums[i] == expectedNums[i];
	}

If all assertions pass, then your solution will be accepted.



Example 1:

	Input: nums = [1,1,2]
	Output: 2, nums = [1,2,_]
	Explanation: Your function should return k = 2, with the first two elements of nums being 1 and 2 respectively.
	It does not matter what you leave beyond the returned k (hence they are underscores).

	
Example 2:

	Input: nums = [0,0,1,1,1,2,2,3,3,4]
	Output: 5, nums = [0,1,2,3,4,_,_,_,_,_]
	Explanation: Your function should return k = 5, with the first five elements of nums being 0, 1, 2, 3, and 4 respectively.
	It does not matter what you leave beyond the returned k (hence they are underscores).



Note:

	0 <= nums.length <= 3 * 10^4
	-100 <= nums[i] <= 100
	nums is sorted in non-decreasing order.


### 解析

根据题意，就是将 nums 列表中的重复元素都去掉，并且保证元素之间的相对位置不发生变化，并且要求不适用额外的空间，只在原列表 nums 中进行操作，最后返回剩余的元素个数。我们可以使用指针的方法，初始化一个索引变量 index 表示指向下一个不同的数字的位置，然后从第 2 个元素开始遍历 nums ，如果当前的元素不等于 nums[index] ，将 index 加一，表示在 index 索引的位置上可以存放不同的元素 nums[index] 。遍历结束，只需要返回 index+1 即可表示元素的个数。


### 解答
				

	class Solution(object):
	    def removeDuplicates(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        if len(nums)==0:return 0
	        index = 0
	        for i in range(1, len(nums)):
	            if nums[i]!=nums[index]:
	                index += 1
	                nums[index] = nums[i]
	        return index+1
            	      
			
### 运行结果

	Runtime: 60 ms, faster than 94.37% of Python online submissions for Remove Duplicates from Sorted Array.
	Memory Usage: 15.1 MB, less than 34.76% of Python online submissions for Remove Duplicates from Sorted Array.

### 解析
另外，因为重复数字都是挨着的，所以如果用变量 count 记录从左往右去掉重复的数字，那么相应的从第二个不重复的数字的索引开始整体要左移 count 个位置，所以解法就是从第二个元素开始遍历 nums ，如果元素 nums[i] 与上一个元素  nums[i-1]  相同，那么 count 加一，如果不相同，说明是不同的数字，可以将 nums[i] 赋给 nums[i-count]  ，遍历结束之后，count 记录了有多少无用的重复数字的个数，使用 len(nums)-count 即可算出不重复的数字的个数。

### 解答	
	class Solution(object):
	    def removeDuplicates(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        if len(nums) == 0: return 0
	        count = 0
	        for i in range(1,len(nums)):
	            if nums[i]==nums[i-1]:
	                count += 1
	            else:
	                nums[i-count] = nums[i]
	        return len(nums)-count
### 运行结果	        
	        
	Runtime: 72 ms, faster than 56.21% of Python online submissions for Remove Duplicates from Sorted Array.
	Memory Usage: 14.7 MB, less than 71.35% of Python online submissions for Remove Duplicates from Sorted Array.    

原题链接：https://leetcode.com/problems/remove-duplicates-from-sorted-array/



您的支持是我最大的动力
