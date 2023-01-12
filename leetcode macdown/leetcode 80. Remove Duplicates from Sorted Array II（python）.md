leetcode  80. Remove Duplicates from Sorted Array II（python）




### 描述



Given an integer array nums sorted in non-decreasing order, remove some duplicates in-place such that each unique element appears at most twice. The relative order of the elements should be kept the same.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

Custom Judge:

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

	Input: nums = [1,1,1,2,2,3]
	Output: 5, nums = [1,1,2,2,3,_]
	Explanation: Your function should return k = 5, with the first five elements of nums being 1, 1, 2, 2 and 3 respectively.
	It does not matter what you leave beyond the returned k (hence they are underscores).

	




Note:

	1 <= nums.length <= 3 * 10^4
	-10^4 <= nums[i] <= 10^4
	nums is sorted in non-decreasing order.



### 解析

根据题意，给定一个按非递减顺序排序的整数数组 nums，就地删除一些重复项，以使每个唯一元素最多出现两次。 元素的相对顺序应保持不变。由于在某些编程语言中无法更改数组的长度，因此必须将结果放在数组 nums 的第一部分。 更正式地说，如果删除重复项后有 k 个元素，则 nums 的前 k 个元素应该保存最终结果。 在前 k 个元素之外留下什么并不重要。将最终结果放入 nums 的前 k 个位置后返回 k。不要为分配额外的空间，题目要求必须通过使用 O(1) 空间复杂度修改输入数组来实现题意。

其实这道题看起来复杂，题目篇幅很多，题意其实很明白，而且实现起来也很简单，基本就是一次遍历就可以解决的事情。我们从左向右遍历 nums 每个元素 c ，同时定义一个索引 index ，初始化为 0 ，因为题目要求我们在 nums 上直接进行修改，所以我们要知道需要修改值的位置索引。当 index < 2 or c > nums[index-2] 的时候，我们就让 nums[index] = c ，同时 index 加一，遍历结束，开头部分的 k 个位置的结果就是我们的答案。


### 解答
				
	class Solution(object):
	    def removeDuplicates(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        index = 0
	        for c in nums:
	            if index < 2 or c > nums[index-2]:
	                nums[index] = c
	                index += 1
	        return index
	            

            	      
			
### 运行结果


	Runtime: 67 ms, faster than 26.95% of Python online submissions for Remove Duplicates from Sorted Array II.
	Memory Usage: 13.5 MB, less than 29.69% of Python online submissions for Remove Duplicates from Sorted Array II.

### 原题链接

https://leetcode.com/problems/remove-duplicates-from-sorted-array-ii/



您的支持是我最大的动力
