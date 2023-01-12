leetcode  482. License Key Formatting（python）



### 描述

You are given a license key represented as a string s that consists of only alphanumeric characters and dashes. The string is separated into n + 1 groups by n dashes. You are also given an integer k.

We want to reformat the string s such that each group contains exactly k characters, except for the first group, which could be shorter than k but still must contain at least one character. Furthermore, there must be a dash inserted between two groups, and you should convert all lowercase letters to uppercase.

Return the reformatted license key.


Example 1:

	Input: s = "5F3Z-2e-9-w", k = 4
	Output: "5F3Z-2E9W"
	Explanation: The string s has been split into two parts, each part has 4 characters.
	Note that the two extra dashes are not needed and can be removed.	
	
Example 2:

	Input: s = "2-5g-3-J", k = 2
	Output: "2-5G-3J"
	Explanation: The string s has been split into three parts, each part has 2 characters except the first part as it could be shorter as mentioned above.


Note:

	1 <= s.length <= 10^5
	s consists of English letters, digits, and dashes '-'.
	1 <= k <= 10^4

### 解析

根据题意，就是将给定的 s ，转换成用短线连接的大写字符串，并且按照顺序，除了第一部分之外，其他每个部分都是要分 k 个字符。其实比较简单，先将 s 去除掉短线，然后转换成大写的字符串，此时的 N 表示 s 的长度。想形成结果只有两种情况：

* N 如果能整除 k ，结果的第一部分就是空字符串，后面的字符串直接按照每部分 k 个字符用短线连接即可。如例子 1 中所示的情况。

* N 如果不能整除 k ，结果的第一个部分就是余数长度的字符串 s[:N%k] ，此时的 s[N%k:] 字符串的长度肯定能整除 k ，也就是后面的字符串肯定能每 k 个字符组成一部分，用短线连接即可。


### 解答
				
	class Solution(object):
	    def licenseKeyFormatting(self, s, k):
	        """
	        :type s: str
	        :type k: int
	        :rtype: str
	        """
	        s = s.replace("-", "").upper()
	        N = len(s)
	        pre = s[:N % k]
	        s = s[N % k:]
	        N = len(s)
	        return (pre + "-" + "-".join([s[i:i + k] for i in range(0, N, k)])).strip("-")
            	      
			
### 运行结果

	Runtime: 36 ms, faster than 79.13% of Python online submissions for License Key Formatting.
	Memory Usage: 15.9 MB, less than 29.85% of Python online submissions for License Key Formatting.

原题链接：https://leetcode.com/problems/license-key-formatting


您的支持是我最大的动力
