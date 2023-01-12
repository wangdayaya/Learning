leetcode  1566. Detect Pattern of Length M Repeated K or More Times（python）

### 描述

Given an array of positive integers arr,  find a pattern of length m that is repeated k or more times.
	
A pattern is a subarray (consecutive sub-sequence) that consists of one or more values, repeated multiple times consecutively without overlapping. A pattern is defined by its length and the number of repetitions.
	
Return true if there exists a pattern of length m that is repeated k or more times, otherwise return false.



Example 1:


	Input: arr = [1,2,4,4,4,4], m = 1, k = 3
	Output: true
	Explanation: The pattern (4) of length 1 is repeated 4 consecutive times. Notice that pattern can be repeated k or more times but not less.
	
Example 2:


	Input: arr = [1,2,1,2,1,1,1,3], m = 2, k = 2
	Output: true
	Explanation: The pattern (1,2) of length 2 is repeated 2 consecutive times. Another valid pattern (2,1) is also repeated 2 times.

Example 3:


	Input: arr = [1,2,1,2,1,3], m = 2, k = 3
	Output: false
	Explanation: The pattern (1,2) is of length 2 but is repeated only 2 times. There is no pattern of length 2 that is repeated 3 or more times.
	
Example 4:


	Input: arr = [1,2,3,1,2], m = 2, k = 2
	Output: false
	Explanation: Notice that the pattern (1,2) exists twice but not consecutively, so it doesn't count.
	
Example 5:


	Input: arr = [2,2,2,2], m = 2, k = 3
	Output: false
	Explanation: The only pattern of length 2 is (2,2) however it's repeated only twice. Notice that we do not count overlapping repetitions.

Note:


	2 <= arr.length <= 100
	1 <= arr[i] <= 100
	1 <= m <= 100
	2 <= k <= 100

### 解析

根据题意，就是给出了一个整数数组 arr ，判断是否有长度为 m 的模式连着重复至少 k 次。这里看英文题目描述可能不是很清晰，但是看了例子就基本知道了，主要记住这 k 个模式不要断开，但也不能互相交叉覆盖。思路比较简单：

* 如果 m*k 大于 arr 的长度，说明无法满足题意，直接返回 False
* 初始化 idx 为 0 表示模式的起始索引，length 为 m\*k 表示模式的长度
* 在 while 循环中，arr[idx:idx+length] 提取当前的子数组，然后判断其是否有满足题意的模式，如果有则直接返回 True ，如果没有则 idx 加一，判断以下一个索引为开头的子数组是否有满足题意的模式
* 当 idx+length>len(arr) 则遍历结束，返回 False 


### 解答
				

	class Solution(object):
	    def containsPattern(self, arr, m, k):
	        """
	        :type arr: List[int]
	        :type m: int
	        :type k: int
	        :rtype: bool
	        """
	        if len(arr) < m*k:
	            return False
	        idx = 0
	        length = m*k
	        while idx+length<=len(arr):
	            subarr = arr[idx:idx+length]
	            if subarr[:m] * k == subarr:
	                return True
	            idx += 1
	        return False
            	      
			
### 运行结果


	Runtime: 20 ms, faster than 84.21% of Python online submissions for Detect Pattern of Length M Repeated K or More Times.
	Memory Usage: 13.5 MB, less than 38.60% of Python online submissions for Detect Pattern of Length M Repeated K or More Times.

原题链接：https://leetcode.com/problems/detect-pattern-of-length-m-repeated-k-or-more-times/



您的支持是我最大的动力
