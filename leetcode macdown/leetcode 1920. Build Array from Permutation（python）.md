leetcode  1920. Build Array from Permutation（python）

### 描述

Given a zero-based permutation nums (0-indexed), build an array ans of the same length where ans[i] = nums[nums[i]] for each 0 <= i < nums.length and return it.

A zero-based permutation nums is an array of distinct integers from 0 to nums.length - 1 (inclusive).



Example 1:

	
	Input: nums = [0,2,1,5,3,4]
	Output: [0,1,2,4,5,3]
	Explanation: The array ans is built as follows: 
	ans = [nums[nums[0]], nums[nums[1]], nums[nums[2]], nums[nums[3]], nums[nums[4]], nums[nums[5]]]
	    = [nums[0], nums[2], nums[1], nums[5], nums[3], nums[4]]
	    = [0,1,2,4,5,3]
	
Example 2:

	Input: nums = [5,0,1,2,3,4]
	Output: [4,5,0,1,2,3]
	Explanation: The array ans is built as follows:
	ans = [nums[nums[0]], nums[nums[1]], nums[nums[2]], nums[nums[3]], nums[nums[4]], nums[nums[5]]]
	    = [nums[5], nums[0], nums[1], nums[2], nums[3], nums[4]]
	    = [4,5,0,1,2,3]




Note:

	1 <= nums.length <= 1000
	0 <= nums[i] < nums.length
	The elements in nums are distinct.


### 解析


根据题意，就是给出了一个整数数组 nums ，然后要将其重新进行排列，排列的规则就是在结果列表中第 i 个索引的位置上的值为 nums[nums[i]] ，所以思路比较简单，就是遍历 nums ，然后将每个 nums[num] 追加到结果 result 中，遍历结束即可得到最终的答案。

### 解答
				

	class Solution(object):
	    def buildArray(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        result = []
	        for num in nums:
	            result.append(nums[num])
	        return result
            	      
			
### 运行结果
	Runtime: 104 ms, faster than 65.36% of Python online submissions for Build Array from Permutation.
	Memory Usage: 13.5 MB, less than 99.51% of Python online submissions for Build Array from Permutation.

### 解析

还有一种方法就是用到了 numpy 中的技巧，直接用 nums 本身作为索引，然后通过 
np.array(nums)[nums] 的方式即可得到答案。按照惯例，这种方法虽然简单，但是不推荐。这里我对 numpy 中的数组技巧作一下简单的说明，代码如下：

例一：

	import numpy as np
	a = np.array([5,2,4,3])  
	print(a[[1,2]]) 

打印：

	 [2 4]
例二：	 

	import numpy as np
	a = np.array([5,2,4,3])  
	print(a[[0,1]])
打印：

	[5 2]

先将索引列表传入入 numpy 所构造的数组中，即可得到新的数组。

从结果来看，这种方法的速度也是很慢的，而且占用的内存较大。

### 解答

	class Solution(object):
	    def buildArray(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        import numpy as np
	        return np.array(nums)[nums]

### 运行结果
	
	Runtime: 144 ms, faster than 19.28% of Python online submissions for Build Array from Permutation.
	Memory Usage: 26 MB, less than 9.80% of Python online submissions for Build Array from Permutation.


原题链接：https://leetcode.com/problems/build-array-from-permutation/



您的支持是我最大的动力
