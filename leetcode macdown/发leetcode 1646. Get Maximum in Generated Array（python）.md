leetcode  1646. Get Maximum in Generated Array（python）

### 描述

You are given an integer n. An array nums of length n + 1 is generated in the following way:

* nums[0] = 0
* nums[1] = 1
* nums[2 * i] = nums[i] when 2 <= 2 * i <= n
* nums[2 * i + 1] = nums[i] + nums[i + 1] when 2 <= 2 * i + 1 <= n

Return the maximum integer in the array nums​​​.



Example 1:

	Input: n = 7
	Output: 3
	Explanation: According to the given rules:
	  nums[0] = 0
	  nums[1] = 1
	  nums[(1 * 2) = 2] = nums[1] = 1
	  nums[(1 * 2) + 1 = 3] = nums[1] + nums[2] = 1 + 1 = 2
	  nums[(2 * 2) = 4] = nums[2] = 1
	  nums[(2 * 2) + 1 = 5] = nums[2] + nums[3] = 1 + 2 = 3
	  nums[(3 * 2) = 6] = nums[3] = 2
	  nums[(3 * 2) + 1 = 7] = nums[3] + nums[4] = 2 + 1 = 3
	Hence, nums = [0,1,1,2,1,3,2,3], and the maximum is 3.

	
Example 2:


	Input: n = 2
	Output: 1
	Explanation: According to the given rules, the maximum between nums[0], nums[1], and nums[2] is 1.

Example 3:

	Input: n = 3
	Output: 2
	Explanation: According to the given rules, the maximum between nums[0], nums[1], nums[2], and nums[3] is 2.





Note:

	0 <= n <= 100


### 解析


根据题意，就是给出一个数字 n ，生成一个 n+1 个数字的列表，找出最大值。生成列表 nums 的过程如下：

* 当索引为 0 的时候，该位置的值为 0
* 当索引为 1 的时候，该位置的值为 1
* 当索引为偶数 i 的时候，该位置的值为 nums[i//2]
* 当索引为奇数 i 的时候，该位置的值为 nums[i//2]+nums[i//2+1]



### 解答
				

	class Solution(object):
	    def getMaximumGenerated(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        if n==0:
	            return 0
	        if n==1:
	            return 1
	        nums = [0, 1]
	        for i in range(2, n+1):
	            if i%2==0:
	                nums.append(nums[i//2])
	            else:
	                nums.append(nums[i//2]+nums[i//2+1])
	        return max(nums)
            	      
			
### 运行结果

	Runtime: 16 ms, faster than 84.09% of Python online submissions for Get Maximum in Generated Array.
	Memory Usage: 13.2 MB, less than 85.23% of Python online submissions for Get Maximum in Generated Array.

### 解析

另外，可以用递归的方式解决。思路和上面一样。递归的出口就是当 n<=1 的时候返回结果，否则就将 n//2 得到索引位置 p ，当 n 为奇数则求 cal(p)+cal(p+1) ，当 n 为偶数则求 cal(p) ，函数递归结束可得到结果。每个元素通过递归结束得到的结果组成的列表，再通过 max 函数求最大值。
### 解答

	class Solution(object):
	    def getMaximumGenerated(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        def cal(n):
	            if n <= 1: return n
	            p = n // 2
	            return cal(p) + cal(p + 1) if n%2 else cal(p)
	        return max(cal(i) for i in range(n + 1))

### 运行结果

	Runtime: 36 ms, faster than 5.42% of Python online submissions for Get Maximum in Generated Array.
	Memory Usage: 13.4 MB, less than 33.73% of Python online submissions for Get Maximum in Generated Array.

原题链接：https://leetcode.com/problems/get-maximum-in-generated-array/



您的支持是我最大的动力
