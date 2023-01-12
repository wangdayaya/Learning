leetcode  2317. Maximum XOR After Operations（python）



### 描述

You are given a 0-indexed integer array nums. In one operation, select any non-negative integer x and an index i, then update nums[i] to be equal to nums[i] AND (nums[i] XOR x).

Note that AND is the bitwise AND operation and XOR is the bitwise XOR operation.

Return the maximum possible bitwise XOR of all elements of nums after applying the operation any number of times.



Example 1:

	Input: nums = [3,2,4,6]
	Output: 7
	Explanation: Apply the operation with x = 4 and i = 3, num[3] = 6 AND (6 XOR 4) = 6 AND 2 = 2.
	Now, nums = [3, 2, 4, 2] and the bitwise XOR of all the elements = 3 XOR 2 XOR 4 XOR 2 = 7.
	It can be shown that 7 is the maximum possible bitwise XOR.
	Note that other operations may be used to achieve a bitwise XOR of 7.

	
Example 2:

	Input: nums = [1,2,3,9,2]
	Output: 11
	Explanation: Apply the operation zero times.
	The bitwise XOR of all the elements = 1 XOR 2 XOR 3 XOR 9 XOR 2 = 11.
	It can be shown that 11 is the maximum possible bitwise XOR.




Note:

	1 <= nums.length <= 10^5
	0 <= nums[i] <= 10^8


### 解析

根据题意，给定一个 0 索引的整数数组 nums 。 在一个操作中，选择任何非负整数 x 和索引 i ，然后将 nums[i] 更新为等于 nums[i] AND (nums[i] XOR x) 。需要注意的是 AND 是按位 AND 运算，XOR 是按位 XOR 运算。在应用该操作任意次数后，返回 nums 的所有元素的最大可能按位异或结果。

其实这道题一开始看有点懵，但是在纸上画一下就好了，目标是所有 nums 中元素的按位异或的最大值，其实 x 的相关操作就是可有可无的，因为肯定存在这么个数（或者没有这个数）可以使最后的结果最大，我们只需要关注什么样的形式可以使结果最大即可。

* 题目中的先进行的 XOR 运算其实就是说可以将 nums[i] 变成任意非负整数
* 然后又进行的 AND 运算其实就是说将 nums[i] 可以将二进制中有 1 的地方变成 1 或者有 1 的地方变成 0 ，但是无法将 0 变为 1 ，这一步就是对 nums[i]  的二进制调整
* 我们再将 nums 中所有元素变成二进制，如下写在纸上按位纵向对齐（例子二），然后找出对应位置所有 1 的个数，要想使最后的所有元素的 XOR 结果最大，那么就是某一个对应的位置的 1 的个数应该为奇数，如果 1 的个数为偶数，我们可以通过上面的操作来将个数减一，换句话说就是只要 1 的个数大于 0 那么这一个位上进行 XOR 操作就是 1 ；如果 1 的个数为 0 ，上面分析我们知道 0 无法变为 1 所以该位置的 1 的个数仍然是 0 ，此时我们进行所有元素的 XOR 运算，其实结果和对原始 nums 中所有元素的 XOR 运算结果一样。

		0001
		0010
		0011
		1001
		0010
		---
		1011

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答
				

	class Solution(object):
	    def maximumXOR(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        L = [bin(n)[2:] for n in nums]
	        count = [0] * 32
	        for s in L:
	            for i,c in enumerate(s):
	                if c == '1':
	                    count[32-(len(s)-i-1)-1] += 1
	        while count and count[0] == 0:
	            count.pop(0)
	        if not count:
	            return 0
	        result = ''
	        for c in count:
	            if c != 0:
	                result += '1'
	            else:
	                result += '0'
	        return int(result,2)
            	      
		

	        
### 运行结果

	
	72 / 72 test cases passed.
	Status: Accepted
	Runtime: 1544 ms
	Memory Usage: 27.1 MB

### 解答

	class Solution(object):
	    def maximumXOR(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        for n in nums:
	            result |= n
	        return result
	        

### 运行结果
	
	72 / 72 test cases passed.
	Status: Accepted
	Runtime: 741 ms
	Memory Usage: 21.5 MB  

### 原题链接

https://leetcode.com/contest/biweekly-contest-81/problems/maximum-xor-after-operations/

您的支持是我最大的动力
