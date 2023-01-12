leetcode  1461. Check If a String Contains All Binary Codes of Size K（python）



### 描述

Given a binary string s and an integer k, return true if every binary code of length k is a substring of s. Otherwise, return false.



Example 1:

	Input: s = "00110110", k = 2
	Output: true
	Explanation: The binary codes of length 2 are "00", "01", "10" and "11". They can be all found as substrings at indices 0, 1, 3 and 2 respectively.

	
Example 2:

	Input: s = "0110", k = 1
	Output: true
	Explanation: The binary codes of length 1 are "0" and "1", it is clear that both exist as a substring. 


Example 3:

	Input: s = "0110", k = 2
	Output: false
	Explanation: The binary code "00" is of length 2 and does not exist in the array.

	


Note:

	1 <= s.length <= 5 * 10^5
	s[i] is either '0' or '1'.
	1 <= k <= 20


### 解析

根据题意，给定一个二进制字符串 s 和一个整数 k ，如果每个长度为 k 的二进制代码都是 s 的子字符串，则返回 true 。 否则，返回 false 。

其实我们按照常规的暴力解法第一个循环找出所有的 k 长度的二进制字符串，第二个循环判断其是否在 s 中，但是这样明显是超时的，因为时间复杂度为 O(N^2) 。我们可以换个角度去思考这个问题反正我们已经知道了要找的子字符串长度为 K ，而且肯定有 2^K 个不同的子字符串，我们只要从左到右遍历 s 中每个长度为 K 的子字符串并将其放入集合 used 中，最后只要集合长度为 2^K 个说明子字符串都找到了返回 True ，否则返回 False 。

时间复杂度为 O(N\*k) ，遍历 s 是 O(N) ，集合去重计算子字符串 hash 是 O(k) ，空间复杂度为 O(N) 。


### 解答
				

	class Solution(object):
	    def hasAllCodes(self, s, k):
	        """
	        :type s: str
	        :type k: int
	        :rtype: bool
	        """
	        count = 1 << k
	        used = set()
	        for i in range(k, len(s) + 1):
	            tmp = s[i-k:i]
	            if tmp not in used:
	                used.add(tmp)
	                count -= 1
	                if count == 0:
	                    return True
	        return False
            	      
			
### 运行结果

	Runtime: 387 ms, faster than 48.15% of Python online submissions for Check If a String Contains All Binary Codes of Size K.
	Memory Usage: 45 MB, less than 48.15% of Python online submissions for Check If a String Contains All Binary Codes of Size K.


### 原题链接

https://leetcode.com/problems/check-if-a-string-contains-all-binary-codes-of-size-k/

您的支持是我最大的动力
