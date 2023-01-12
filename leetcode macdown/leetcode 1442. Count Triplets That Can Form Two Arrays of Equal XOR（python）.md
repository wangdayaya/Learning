leetcode  1442. Count Triplets That Can Form Two Arrays of Equal XOR（python）

### 描述

Given an array of integers arr.

We want to select three indices i, j and k where (0 <= i < j <= k < arr.length).

Let's define a and b as follows:

* a = arr[i] ^ arr[i + 1] ^ ... ^ arr[j - 1]
* b = arr[j] ^ arr[j + 1] ^ ... ^ arr[k]

Note that ^ denotes the bitwise-xor operation.

Return the number of triplets (i, j and k) Where a == b.



Example 1:

	Input: arr = [2,3,1,6,7]
	Output: 4
	Explanation: The triplets are (0,1,2), (0,2,2), (2,3,4) and (2,4,4)


	
Example 2:

	Input: arr = [1,1,1,1,1]
	Output: 10


Example 3:


	Input: arr = [2,3]
	Output: 0
	
Example 4:


	Input: arr = [1,3,5,7,9]
	Output: 3
	
Example 5:

	Input: arr = [7,11,12,9,5,2,7,17,22]
	Output: 8


Note:


	1 <= arr.length <= 300
	1 <= arr[i] <= 10^8

### 解析


根据题意，就是要从整数列表 arr 中取出三个索引 i 、j 、k ，且要求 0 <= i < j <= k < arr.length ，并且计算两个值 a 和 b  ：

	a = arr[i] ^ arr[i + 1] ^ ... ^ arr[j - 1]
	b = arr[j] ^ arr[j + 1] ^ ... ^ arr[k]

其中的 ^ 就是异或运算，然后计算有多少对的 i 、j 、k 能满足 a==b 。

最简单最直接的办法肯定是暴力求解了，直接使用三重循环找出所有可能的 i 、j 、k ，然后计算  arr[i:j] 和 arr[j:k + 1] 的运算结果是否相等，如果相等计数器加一，三重循环结束返回最后的计数器结果 result 。

结果可想而知是运行超时，因为题目中的 arr 长度太大，且每个元素的值也很大。肯定还是要找规律解题的。


### 解答
				
	class Solution(object):
	    def countTriplets(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: int
	        """
	        if len(arr) < 2: return 0
	        if len(arr) == 2: return 1 if arr[0] == arr[1] else 0
	
	        def xor(nums):
	            result = 1
	            for i in nums:
	                result ^= i
	            return result
	
	        N = len(arr)
	        result = 0
	        for i in range(N - 1):
	            for j in range(i + 1, N):
	                for k in range(j, N):
	                    if xor(arr[i:j]) == xor(arr[j:k + 1]):
	                        result += 1
	        return result
	                        
	                

            	      
			
### 运行结果

	Time Limit Exceeded

### 解析


其实可以用到抑或运算的性质，既然题目中给出了 a 要和 b 的异或结果相等的要求，那么根据异或运算相同为 0 不同为 1 的法则，可以知道 a^b == 0 ，所以题目中要找的就是满足 a^b == 0 的三元组 (i , j , k) 。

另外，a 和 b 是由 1 个或者多个数字进行抑或运算得到的结果，而异或运算是有结合律的，也就是 ( a ^ b ) ^ c = a ^ ( b ^ c ) 的，所以只要我们发现 a 到 c 的列表中的数字的异或结果为 0 ，a 的位置为 i ，c 的位置为 k ，中间的 b 也就是 j ，是在满足 0 <= i < j <= k < arr.length 的条件下是可以任意移动的，有 c-a 种可能。

根据这个性质，使用双重循环来解题即可得到答案。

### 解答

	class Solution(object):
	    def countTriplets(self, arr):
	        """
	        :type arr: List[int]
	        :rtype: int
	        """
	        result = 0
	        for i in range(len(arr) - 1):
	            xor = 0
	            for j in range(i, len(arr)):
	                xor ^= arr[j]
	                if xor == 0:
	                    result += j - i
	        return result


### 运行结果

	
	Runtime: 43 ms, faster than 88.24% of Python online submissions for Count Triplets That Can Form Two Arrays of Equal XOR.
	Memory Usage: 13.5 MB, less than 47.06% of Python online submissions for Count Triplets That Can Form Two Arrays of Equal XOR.

原题链接：https://leetcode.com/problems/count-triplets-that-can-form-two-arrays-of-equal-xor/



您的支持是我最大的动力
