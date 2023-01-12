本文正在参加「Python主题月」，详情查看 [活动链接](https://juejin.cn/post/6979532761954533390/ "https://juejin.cn/post/6979532761954533390/")
### 描述


Given an integer array nums and an integer val, remove all occurrences of val in nums in-place. The relative order of the elements may be changed.

Since it is impossible to change the length of the array in some languages, you must instead have the result be placed in the first part of the array nums. More formally, if there are k elements after removing the duplicates, then the first k elements of nums should hold the final result. It does not matter what you leave beyond the first k elements.

Return k after placing the final result in the first k slots of nums.

Do not allocate extra space for another array. You must do this by modifying the input array in-place with O(1) extra memory.

**Custom Judge**:

The judge will test your solution with the following code:

	int[] nums = [...]; // Input array
	int val = ...; // Value to remove
	int[] expectedNums = [...]; // The expected answer with correct length.
	                            // It is sorted with no values equaling val.
	
	int k = removeElement(nums, val); // Calls your implementation
	
	assert k == expectedNums.length;
	sort(nums, 0, k); // Sort the first k elements of nums
	for (int i = 0; i < actualLength; i++) {
	    assert nums[i] == expectedNums[i];
	}
If all assertions pass, then your solution will be accepted.

Example 1:

	Input: nums = [3,2,2,3], val = 3
	Output: 2, nums = [2,2,_,_]
	Explanation: Your function should return k = 2, with the first two elements of nums being 2.
	It does not matter what you leave beyond the returned k (hence they are underscores).

	
Example 2:

	Input: nums = [0,1,2,2,3,0,4,2], val = 2
	Output: 5, nums = [0,1,4,0,3,_,_,_]
	Explanation: Your function should return k = 5, with the first five elements of nums containing 0, 0, 1, 3, and 4.
	Note that the five elements can be returned in any order.
	It does not matter what you leave beyond the returned k (hence they are underscores).




Note:

	0 <= nums.length <= 100
	0 <= nums[i] <= 50
	0 <= val <= 100


### 解析

这个题和[第 26 题](https://leetcode.com/problems/remove-duplicates-from-sorted-array/)很像，但稍有不同，根据题意，这道题是去掉 nums 列表中的指定元素 val ，然后在原列表上面进行修改，不适用额外的空间，将剩下的非 val 的元素都放到 nums 的前几个位置，并返回剩下元素的个数。思路和之前的 26 题很像：

- 初始化一个计数器 count 记录有多少个需要去掉的 val 
- 遍历 nums 中的所有元素，如果当前元素 nums[i] 等于 val ，则计数器 count 加一，如果不相等，则将 nums[i] 赋给 nums[i-count] ，这里就相当于是将合法的其他元素整体前移了 count 个位置
- 遍历结束，返回 len(nums)-count 为去掉所有 val 之后的元素个数。


### 解答
				
	class Solution(object):
	    def removeElement(self, nums, val):
	        """
	        :type nums: List[int]
	        :type val: int
	        :rtype: int
	        """
	        if len(nums) == 0: return  0
	        count = 0
	        for i in range(len(nums)):
	            if nums[i] == val:
	                count += 1
	            else:
	                nums[i-count] = nums[i]
	        return len(nums)-count

            	      
			
### 运行结果

	Runtime: 12 ms, faster than 99.23% of Python online submissions for Remove Element.
	Memory Usage: 13.6 MB, less than 11.06% of Python online submissions for Remove Element.

### 解析

另外还有一种思路：

- 初始化 count 来表示非 val 元素的索引
- 在遍历 nums 的时候，如果 nums[i] 不等于 val ，那么就将 nums[i] 赋给 nums[count] ，并 count 加一，这样 nums 中的非 val 元素就会挨个按照 count 的自增而放逐个到 nums 的前面的位置
- 遍历结束，这时的 count 即为前面的非 val 的元素个数。


### 解答


	class Solution(object):
	    def removeElement(self, nums, val):
	        """
	        :type nums: List[int]
	        :type val: int
	        :rtype: int
	        """
	        if len(nums) == 0: return  0
	        count = 0
	        for i in range(len(nums)):
	            if nums[i]!=val:
	                nums[count] = nums[i]
	                count+=1
	        return count

### 运行结果

	Runtime: 24 ms, faster than 44.59% of Python online submissions for Remove Element.
	Memory Usage: 13.5 MB, less than 36.04% of Python online submissions for Remove Element.

原题链接：https://leetcode.com/problems/remove-element/



您的支持是我最大的动力
