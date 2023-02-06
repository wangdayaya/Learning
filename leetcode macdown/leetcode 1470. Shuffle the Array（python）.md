leetcode  1470. Shuffle the Array（python）




### 描述


Given the array nums consisting of 2n elements in the form [x1,x2,...,xn,y1,y2,...,yn].

Return the array in the form [x1,y1,x2,y2,...,xn,yn].




Example 1:

	Input: nums = [2,5,1,3,4,7], n = 3
	Output: [2,3,5,4,1,7] 
	Explanation: Since x1=2, x2=5, x3=1, y1=3, y2=4, y3=7 then the answer is [2,3,5,4,1,7].
	
Example 2:


	Input: nums = [1,2,3,4,4,3,2,1], n = 4
	Output: [1,4,2,3,3,2,4,1]

Example 3:

	Input: nums = [1,1,2,2], n = 2
	Output: [1,2,1,2]



Note:


	1 <= n <= 500
	nums.length == 2n
	1 <= nums[i] <= 10^3

### 解析

给定一个数组 nums ，数组中有 2n 个元素，按 [x1,x2,...,xn,y1,y2,...,yn] 的格式排列。要求我们将数组按 [x1,y1,x2,y2,...,xn,yn] 格式重新排列，返回重排后的数组。

这道题其实就是一道 Easy 的简单题，我们用最简单的方法，直接定义一个新的数组 result ，然后从开头开始找元素，每次把 i 和 i+n 的元素加入到 result 即可，i 不断自增，最后肯定会将 nums 按照题意存放到 result 中。

时间复杂度为 O(N) ，空间复杂度为 O(N)。

### 解答

	class Solution(object):
	    def shuffle(self, nums, n):
	        """
	        :type nums: List[int]
	        :type n: int
	        :rtype: List[int]
	        """
	        result = []
	        for i in range(n):
	            result.append(nums[i])
	            result.append(nums[i+n])
	        return result

### 运行结果

	Runtime 41 ms，Beats 77.80%
	Memory 13.5 MB，Beats 91.84%

### 原题链接

https://leetcode.com/problems/shuffle-the-array/description/


您的支持是我最大的动力
