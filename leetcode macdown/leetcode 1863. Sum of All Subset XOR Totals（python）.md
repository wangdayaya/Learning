leetcode  1863. Sum of All Subset XOR Totals（python）

### 描述

The XOR total of an array is defined as the bitwise XOR of all its elements, or 0 if the array is empty.

* For example, the XOR total of the array [2,5,6] is 2 XOR 5 XOR 6 = 1.

Given an array nums, return the sum of all XOR totals for every subset of nums. 

Note: Subsets with the same elements should be counted multiple times.

An array a is a subset of an array b if a can be obtained from b by deleting some (possibly zero) elements of b.



Example 1:

	Input: nums = [1,3]
	Output: 6
	Explanation: The 4 subsets of [1,3] are:
	- The empty subset has an XOR total of 0.
	- [1] has an XOR total of 1.
	- [3] has an XOR total of 3.
	- [1,3] has an XOR total of 1 XOR 3 = 2.
	0 + 1 + 3 + 2 = 6

	
Example 2:

	Input: nums = [5,1,6]
	Output: 28
	Explanation: The 8 subsets of [5,1,6] are:
	- The empty subset has an XOR total of 0.
	- [5] has an XOR total of 5.
	- [1] has an XOR total of 1.
	- [6] has an XOR total of 6.
	- [5,1] has an XOR total of 5 XOR 1 = 4.
	- [5,6] has an XOR total of 5 XOR 6 = 3.
	- [1,6] has an XOR total of 1 XOR 6 = 7.
	- [5,1,6] has an XOR total of 5 XOR 1 XOR 6 = 2.
	0 + 5 + 1 + 6 + 4 + 3 + 7 + 2 = 28


Example 3:

	Input: nums = [3,4,5,6,7,8]
	Output: 480
	Explanation: The sum of all XOR totals for every subset is 480.

	


Note:

	1 <= nums.length <= 12
	1 <= nums[i] <= 20


### 解析


根据题意，就是找出所有的 nums 的子集，然后计算出每个子集的元素异或值，将所有的异或值加起来的和。这里用到的内置函数直接找所有的子集，直接进行计算。高手应该要自己求所有子集，而不是学我用内置函数这种低级水平。

### 解答
					
	class Solution(object):
	    def subsetXORSum(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        for i in range(len(nums)+1):
	            for j in itertools.combinations(nums,i):
	                t = 0
	                for k in j:
	                    t ^= k
	                result += t
	        return result
### 运行结果

	Runtime: 56 ms, faster than 68.31% of Python online submissions for Sum of All Subset XOR Totals.
	Memory Usage: 13.2 MB, less than 93.43% of Python online submissions for Sum of All Subset XOR Totals.
	        


### 解析


不用内置函数找 nums 子集，直接通过前向遍历找出所有的子集，然后将不为空的子集进行异或运算，将所有子集进行异或运算之后的结果相加。

### 解答
    
	    
	class Solution(object):
	    def subsetXORSum(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        subsets = [[]]
	        for num in nums:
	            subsets += [subset+[num] for subset in subsets]
	        for subset in subsets:
	            if subset:
	                result += functools.reduce(lambda x,y:x^y,subset)
	        return result          	      
			
### 运行结果
	
	Runtime: 100 ms, faster than 39.20% of Python online submissions for Sum of All Subset XOR Totals.
	Memory Usage: 14 MB, less than 12.21% of Python online submissions for Sum of All Subset XOR Totals.


原题链接：https://leetcode.com/problems/sum-of-all-subset-xor-totals



您的支持是我最大的动力
