leetcode  1784. Check if Binary String Has at Most One Segment of Ones（python）

### 描述

Given a binary string s ​​​​​without leading zeros, return true​​​ if s contains at most one contiguous segment of ones. Otherwise, return false.



Example 1:

	Input: s = "1001"
	Output: false
	Explanation: The ones do not form a contiguous segment.

	
Example 2:

	Input: s = "110"
	Output: true






Note:

	1 <= s.length <= 100
	s[i]​​​​ is either '0' or '1'.
	s[0] is '1'.


### 解析


【一定要好好都题意，我就是没理解透彻，错了 4 回才发现题目的真正含义！】

根据题意，就是判断字符串 s 中是否只有一个 1 组成的子字符串，思路很简单，就是 s.split("0") 获得列表，如果其中的长度大于 0 的元素个数大于 1 个，直接返回 False ，否则返回 True 。

### 解答
				
	
	class Solution(object):
	    def checkOnesSegment(self, s):
	        """
	        :type s: str
	        :rtype: bool
	        """
	        ones = s.split("0")
	        count = 0
	        for one in ones:
	            if len(one)>0:
	                count+=1
	                if count>1:
	                    return False
	        return True
            	      
			
### 运行结果


	Runtime: 20 ms, faster than 39.45% of Python online submissions for Check if Binary String Has at Most One Segment of Ones.
	Memory Usage: 13.4 MB, less than 53.21% of Python online submissions for Check if Binary String Has at Most One Segment of Ones.
	
	
	
### 解析


还有大神这样解答，只要 '01'  字符串不出现在 s 中直接返回 True ，否则返回 False 。真的是巧妙，因为只要有 '01' 出现，肯定字符串是 1 开头的，所以只要出现了 ‘01’ 字符串，肯定至少有两个 1 的组成的子字符串出现。

### 解答
				
	class Solution(object):
	    def checkOnesSegment(self, s):
	        """
	        :type s: str
	        :rtype: bool
	        """
	        return '01' not in s
            	      
			
### 运行结果
	
	Runtime: 8 ms, faster than 98.17% of Python online submissions for Check if Binary String Has at Most One Segment of Ones.
	Memory Usage: 13.4 MB, less than 79.82% of Python online submissions for Check if Binary String Has at Most One Segment of Ones.

原题链接：https://leetcode.com/problems/check-if-binary-string-has-at-most-one-segment-of-ones/



您的支持是我最大的动力
