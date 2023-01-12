leetcode 2357. Make Array Zero by Subtracting Equal Amounts （python）




### 描述

You are given a non-negative integer array nums. In one operation, you must:

* Choose a positive integer x such that x is less than or equal to the smallest non-zero element in nums.
* Subtract x from every positive element in nums.
	
Return the minimum number of operations to make every element in nums equal to 0.



Example 1:

	Input: nums = [1,5,0,3,5]
	Output: 3
	Explanation:
	In the first operation, choose x = 1. Now, nums = [0,4,0,2,4].
	In the second operation, choose x = 2. Now, nums = [0,2,0,0,2].
	In the third operation, choose x = 2. Now, nums = [0,0,0,0,0].

	
Example 2:

	Input: nums = [0]
	Output: 0
	Explanation: Each element in nums is already 0 so no operations are needed.





Note:

	1 <= nums.length <= 100
	0 <= nums[i] <= 100


### 解析

根据题意，给定一个非负整数数组 nums。 在每次操作中都要进行以下步骤：

* 选择一个正整数 x，使得 x 小于或等于 nums 中最小的非零元素。
* 将 nums 中的每个正整数中减去 x。

返回使 nums 中的每个元素都等于 0 的最小操作数。

这道题其实不难，因为每次都要取最小不为 0 的数字并且用他取减 nums 中的每个元素，所以我们可以先将 nums 进行升序排序，然后循环进行以下操作，使用 result 计数，最后得到的 result 就是结果：

* 我们将前面的 0 都弹出去
* 这时候可以拿到最小的不为 0 的数字，将 nums 中的每个元素减去这个数字，此时的 nums 仍然都是升序的顺序
* 重复上述过程直到 nums 为空跳出循环


在写代码的时候有些细节需要注意，时间复杂度为 O(NlogN) ，空间复杂度为 O(1)。

### 解答

	class Solution(object):
	    def minimumOperations(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        nums.sort()
	        while nums:
	            while nums and nums[0] == 0:
	                nums.pop(0)
	            if not nums:
	                break
	            t = nums[0]
	            for i in range(len(nums)):
	                nums[i] -= t
	            result += 1
	        return result

### 运行结果

	95 / 95 test cases passed.
	Status: Accepted
	Runtime: 30 ms
	Memory Usage: 13.2 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-304/problems/make-array-zero-by-subtracting-equal-amounts/

您的支持是我最大的动力
