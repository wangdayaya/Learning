leetcode  2195. Append K Integers With Minimal Sum（python）


### 前言

这是 Weekly Contest 283 的第二题，考察的就是对题目的理解程度，用数学的方法解题即可，难度 Medium 。

### 描述


You are given an integer array nums and an integer k. Append k unique positive integers that do not appear in nums to nums such that the resulting total sum is minimum.

Return the sum of the k integers appended to nums.


Example 1:

	Input: nums = [1,4,25,10,25], k = 2
	Output: 5
	Explanation: The two unique positive integers that do not appear in nums which we append are 2 and 3.
	The resulting sum of nums is 1 + 4 + 25 + 10 + 25 + 2 + 3 = 70, which is the minimum.
	The sum of the two integers appended is 2 + 3 = 5, so we return 5.





Note:

	1 <= nums.length <= 10^5
	1 <= nums[i] <= 10^9
	1 <= k <= 10^8


### 解析


根据题意，给定一个整数数组 nums 和一个整数 k。 将未出现在 nums 中的 k 个唯一正整数加到 nums 中，以使所得总和最小。返回加到 nums 的 k 个整数的总和。

其实用我们最朴素的思想就能知道，要想使得附加的数字的总和最小，那就尽量选择小的数字即可，其实我们在没有时间复杂度限制的情况下，我们从 1 开始遍历自然数 n ，当 n 在 nums 中的时候直接跳过去找下一个数字，如果没有在 nums 中则将其加入到结果总和 result 中，一直这样进行下去直到找到 k 个数字返回 result 即可，这样找出来的 k 个数字总和肯定是最小的。

但是我们这里的 nums 的长度最大为 10^5 ，nums[i] 最大为 10^9 ，所以这种方法肯定是要超时的，我们其实可以这样考虑，将 nums 进行升序排序，假如现在有 nums=[3,5,7]  ，k 可能有以下几种情况：

* 当 k<3 的时候，我们直接返回从 1 到 k 的总和即可，这样可以保证总和 result 最小，这可以通过数学公式算出，我们用一个函数 add 来计算。
* 当 k>=3 的时候，经过上面的一步，此时我们的 k 为 k-2 （因为 1 和 2 已经加入到 result 中了），我们因为 nums 已经排序，我们遍历 nums 中每两个数字之间缺失的自然数个数 n ， 尽可能地找出小的数字来加入到 result 中，如果在这个过程中 k 减少为 0 ，则直接返回 result 。
* 当遍历完 nums 如果 k 仍然大于 0 ，则将 [nums[-1] + 1, nums[-1] + k] 范围的数字加入到 result 中，返回 result 即可。

时间复杂度为 O(N) ，空间复杂度为 O(1) ，因为没有开辟新的空间。

### 解答
				
	class Solution(object):
	    def minimalKSum(self, nums, k):
	        def add(x, y):
	            N = y - x + 1
	            if N % 2:
	                return (x + y) * (N // 2) + x + N // 2
	            return (x + y) * (N // 2)
	        nums.sort()
	        result = 0
	        if nums[0] > k:
	            return add(1, k)
	        result += add(1, nums[0] - 1)
	        k -= (nums[0] - 1)
	        for i in range(1, len(nums)):
	            if nums[i] - nums[i - 1] - 1 <= 0:
	                continue
	            elif k > nums[i] - nums[i - 1] - 1:
	                result += add(nums[i - 1] + 1, nums[i] - 1)
	                k -= (nums[i] - nums[i - 1] - 1)
	            else:
	                result += add(nums[i - 1] + 1, nums[i - 1] + k)
	                return result
	        return result + add(nums[-1] + 1, nums[-1] + k)
            
            
                
                

            	      
			
### 运行结果

	108 / 108 test cases passed.
	Status: Accepted
	Runtime: 799 ms
	Memory Usage: 25.6 MB


### 原题链接


https://leetcode.com/contest/weekly-contest-283/problems/append-k-integers-with-minimal-sum/


您的支持是我最大的动力
