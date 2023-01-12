leetcode  1829. Maximum XOR for Each Query（python）

### 描述

You are given a sorted array nums of n non-negative integers and an integer maximumBit. You want to perform the following query n times:

Find a non-negative integer k < 2<sup>maximumBit</sup> such that nums[0] XOR nums[1] XOR ... XOR nums[nums.length-1] XOR k is maximized. k is the answer to the i<sup>th</sup> query.
Remove the last element from the current array nums.
Return an array answer, where answer[i] is the answer to the i<sup>th</sup> query.

 



Example 1:

	Input: nums = [0,1,1,3], maximumBit = 2
	Output: [0,3,2,3]
	Explanation: The queries are answered as follows:
	1st query: nums = [0,1,1,3], k = 0 since 0 XOR 1 XOR 1 XOR 3 XOR 0 = 3.
	2nd query: nums = [0,1,1], k = 3 since 0 XOR 1 XOR 1 XOR 3 = 3.
	3rd query: nums = [0,1], k = 2 since 0 XOR 1 XOR 2 = 3.
	4th query: nums = [0], k = 3 since 0 XOR 3 = 3.

	
Example 2:
	
	Input: nums = [2,3,4,7], maximumBit = 3
	Output: [5,2,6,5]
	Explanation: The queries are answered as follows:
	1st query: nums = [2,3,4,7], k = 5 since 2 XOR 3 XOR 4 XOR 7 XOR 5 = 7.
	2nd query: nums = [2,3,4], k = 2 since 2 XOR 3 XOR 4 XOR 2 = 7.
	3rd query: nums = [2,3], k = 6 since 2 XOR 3 XOR 6 = 7.
	4th query: nums = [2], k = 5 since 2 XOR 5 = 7.


Example 3:

	Input: nums = [0,1,2,2,5,7], maximumBit = 3
	Output: [4,3,6,4,6,7]

	


Note:


	nums.length == n
	1 <= n <= 10^5
	1 <= maximumBit <= 20
	0 <= nums[i] < 2^maximumBit
	nums​​​ is sorted in ascending order.

### 解析


根据题意，就是给出了一个已经排序的长度为 n 的整数列表 nums ，并且给出了一个整数 maximumBit，让我们操作以下步骤：

* 将前 1 个元素进行抑或运算，并且和  [0, 2^ maximumBit-1] 之间的某个整数进行抑或运算可以保证结果最大，将这个整数加入结果列表中
* 将前 2 个元素进行抑或运算，并且和  [0, 2^ maximumBit-1] 之间的某个整数进行抑或运算可以保证结果最大，将这个整数加入结果列表中
* 将前 3 个元素进行抑或运算，并且和  [0, 2^ maximumBit-1] 之间的某个整数进行抑或运算可以保证结果最大，将这个整数加入结果列表中
* ......
* 前 n 个元素进行抑或运算，并且和  [0, 2^ maximumBit-1] 之间的某个整数进行抑或运算可以保证结果最大，将这个整数加入结果列表中

最后返回结果列表。

最直接的方法就是暴力求解，先将每个位置之前的所有元素都求解出抑或结果，然后双重循环找出每个位置最合适的  [0, 2^ maximumBit-1] 之间个整数结果。但是这种结果超时了。因为题目中的 n 和 maximumBit 都很大。



### 解答
				
	class Solution(object):
	    def getMaximumXor(self, nums, maximumBit):
	        """
	        :type nums: List[int]
	        :type maximumBit: int
	        :rtype: List[int]
	        """
	        XORS = [nums[0]]
	        for i,num in enumerate(nums[1:]):
	            XORS.append(XORS[-1] ^ num)
	        result = [0]*len(nums)
	        for i,num in enumerate(XORS):
	            tmp = num ^ 0
	            for bit in range(1, 2**maximumBit):
	                if num^bit > tmp:
	                    tmp = num^bit
	                    result[i] = bit
	        return result[::-1]

            	      
			
### 运行结果

	Time Limit Exceeded

### 解析

其实这是一道考察抑或运算的题，要解题需要知道两个关键知识点：

* 不管几个 [0, 2^maximumBit-1] 这个范围内的数字进行抑或运算，最大结果也只能是 2^maximumBit -1
* 如果 x ^ y == z，那么 x ^ z == y，同时 y ^ z == x

结合题意我们可以得到:

* nums 中每个元素满足0 <= nums[i] < 2^maximumBit
* k 取值肯定满足抑或结果最大为(2^maximumBit-1)，所以 nums[0] ^ nums[1] ^ ... ^ nums[nums.length-1] ^ k == (2^maximumBit-1)，那么 k == nums[0] ^ nums[1] ^ ... ^ nums[nums.length-1] ^ (2^maximumBit-1) 




### 解答

	class Solution(object):
	    def getMaximumXor(self, nums, maximumBit):
	        """
	        :type nums: List[int]
	        :type maximumBit: int
	        :rtype: List[int]
	        """
	        ALL = 0
	        for num in nums: ALL ^= num
	        maxresult = 2**maximumBit-1
	        result = []
	        while nums:
	            result.append(ALL ^ maxresult)
	            ALL ^= nums.pop()
	        return result 
### 运行结果
	
	Runtime: 1113 ms, faster than 25.93% of Python online submissions for Maximum XOR for Each Query.
	Memory Usage: 29.1 MB, less than 81.48% of Python online submissions for Maximum XOR for Each Query.

原题链接：https://leetcode.com/problems/maximum-xor-for-each-query/



您的支持是我最大的动力
