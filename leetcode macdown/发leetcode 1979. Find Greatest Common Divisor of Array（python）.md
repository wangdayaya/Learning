

### 描述


Given an integer array nums, return the greatest common divisor of the smallest number and largest number in nums.

The greatest common divisor of two numbers is the largest positive integer that evenly divides both numbers.




Example 1:

	Input: nums = [2,5,6,9,10]
	Output: 2
	Explanation:
	The smallest number in nums is 2.
	The largest number in nums is 10.
	The greatest common divisor of 2 and 10 is 2.	
	
Example 2:

	Input: nums = [7,5,6,8,3]
	Output: 1
	Explanation:
	The smallest number in nums is 3.
	The largest number in nums is 8.
	The greatest common divisor of 3 and 8 is 1.

Example 3:

	Input: nums = [3,3]
	Output: 3
	Explanation:
	The smallest number in nums is 3.
	The largest number in nums is 3.
	The greatest common divisor of 3 and 3 is 3.



Note:

	2 <= nums.length <= 1000
	1 <= nums[i] <= 1000


### 解析

根据题意，就是给出了一个整数列表 nums ，找 nums 中最大的数和最小的数的最大共同除数，其实思路很简单：

* 如果使用 set 函数之后的集合长度只有一个，那么就说明数字都一样，直接返回 nums[0]
* 否则就直接找出最小的数 mn ，然后找出最大的数 mx 
* 因为如果有除数肯定最大是 mn ，所以直接从大到小遍历 [mn,1] 的数字 i ，只要判断出来 mn 和 mx 都能被整除，就直接返回 i

### 解答
				
	class Solution(object):
	    def findGCD(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        if set(nums) == 1:
	            return nums[0]
	        mn = min(nums)
	        mx = max(nums)
	        for i in range(mn, 0, -1):
	            if mx%i==0 and mn%i==0:
	                return i
            	      
			
### 运行结果

	Runtime: 36 ms, faster than 95.07% of Python online submissions for Find Greatest Common Divisor of Array.
	Memory Usage: 13.6 MB, less than 27.41% of Python online submissions for Find Greatest Common Divisor of Array.


### 解析

另外还有就是直接使用 python 的内置函数，思路和上面一样，先找出 nums 的最大数和最小数，然后使用内置函数 gcd 即可直接求出答案。

### 解答

	class Solution:
	    def findGCD(self, nums: List[int]) -> int:
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        nums.sort()
	        min=nums[0]
	        max=nums[-1]
	        ans=math.gcd(min,max)
	        return(ans)

### 运行结果

	Runtime: 94 ms, faster than 13.10% of Python3 online submissions for Find Greatest Common Divisor of Array.
	Memory Usage: 14.3 MB, less than 66.94% of Python3 online submissions for Find Greatest Common Divisor of Array.


原题链接：https://leetcode.com/problems/find-greatest-common-divisor-of-array/


您的支持是我最大的动力
