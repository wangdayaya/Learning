leetcode  1796. Second Largest Digit in a String（python）

### 描述

Given an alphanumeric string s, return the second largest numerical digit that appears in s, or -1 if it does not exist.

An alphanumeric string is a string consisting of lowercase English letters and digits.



Example 1:

	Input: s = "dfa12321afd"
	Output: 2
	Explanation: The digits that appear in s are [1, 2, 3]. The second largest digit is 2.	
	
Example 2:

	Input: s = "abc1111"
	Output: -1
	Explanation: The digits that appear in s are [1]. There is no second largest digit. 




Note:

	1 <= s.length <= 500
	s consists of only lowercase English letters and/or digits.


### 解析

根据题意，就是找出字符串中第二大的数字，没有则返回 -1 。如果是纯字母字符串直接返回 -1 ，否则先使用正则提取出所有的单个数字，然后去重之后判断剩余数字的个数，如果个数大于 1 则返回第二大数字，否则直接返回 -1 。

### 解答
				
	class Solution(object):
	    def secondHighest(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        if s.isalpha():
	            return -1
	        r = [int(i) for i in re.findall('[0-9]{1}', s)]
	        r = list(set(r))
	        r.sort()
	        if len(r) >= 2:
	            return r[-2]
	        return -1
            	      
			
### 运行结果

	Runtime: 48 ms, faster than 19.05% of Python online submissions for Second Largest Digit in a String.
	Memory Usage: 13.6 MB, less than 20.95% of Python online submissions for Second Largest Digit in a String.


原题链接：https://leetcode.com/problems/second-largest-digit-in-a-string/


您的支持是我最大的动力