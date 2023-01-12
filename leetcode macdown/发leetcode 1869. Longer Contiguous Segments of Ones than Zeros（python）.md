leetcode  1869. Longer Contiguous Segments of Ones than Zeros（python）

### 描述

Given a binary string s, return true if the longest contiguous segment of 1s is strictly longer than the longest contiguous segment of 0s in s. Return false otherwise.

* For example, in s = "110100010" the longest contiguous segment of 1s has length 2, and the longest contiguous segment of 0s has length 3.

Note that if there are no 0s, then the longest contiguous segment of 0s is considered to have length 0. The same applies if there are no 1s.





Example 1:

	Input: s = "1101"
	Output: true
	Explanation:
	The longest contiguous segment of 1s has length 2: "1101"
	The longest contiguous segment of 0s has length 1: "1101"
	The segment of 1s is longer, so return true.

	
Example 2:


	Input: s = "111000"
	Output: false
	Explanation:
	The longest contiguous segment of 1s has length 3: "111000"
	The longest contiguous segment of 0s has length 3: "111000"
	The segment of 1s is not longer, so return false.

Example 3:


	Input: s = "110100010"
	Output: false
	Explanation:
	The longest contiguous segment of 1s has length 2: "110100010"
	The longest contiguous segment of 0s has length 3: "110100010"
	The segment of 1s is not longer, so return false.
	




Note:

	1 <= s.length <= 100
	s[i] is either '0' or '1'.


### 解析


根据题意，如果 s 中最长的 1 字符串的长度大于最长的 0 字符串，返回 true ，否则返回 false 。 思路比较简单， s.split("0") 可以得到分割出来的包含 1 的字符串列表，s.split("1") 可以得到分割出来的包含 0 的字符串，将两者排序找出各自最大的长度的字符串，判断最长的 1 字符串的长度是否大于最长的 0 字符串的长度。

### 解答
				
	class Solution(object):
	    def checkZeroOnes(self, s):
	        """
	        :type s: str
	        :rtype: bool
	        """
	        one = sorted( s.split("0"), key=lambda x: len(x))[-1]
	        zero = sorted( s.split("1"), key=lambda x: len(x))[-1]
	        if len(one)>len(zero):
	            return True
	        return False
            	      
			
### 运行结果

	Runtime: 16 ms, faster than 89.50% of Python online submissions for Longer Contiguous Segments of Ones than Zeros.
	Memory Usage: 13.6 MB, less than 27.00% of Python online submissions for Longer Contiguous Segments of Ones than Zeros.

### 解析

另外可以直接遍历这个字符串求解：

* 用变量 mx\_1, mx_0 记录全局出现连续 1 的最大个数或者连续 0 的最大个数
* 用变量 tmp\_1, tmp_0 记录局部出现连续 1 的最大个数或者连续 0 的最大个数，当出现前后字符不相等的时候，更新 mx\_1 或者 mx\_0 
* 最后返回 max(mx\_1, tmp\_1) > max(mx\_0, tmp\_0) 即可



### 解答

	class Solution(object):
	    def checkZeroOnes(self, s):
	        """
	        :type s: str
	        :rtype: bool
	        """
	        mx_1, mx_0 = 0, 0
	        tmp_1, tmp_0 = 0, 0
	        p = s[0]
	        if p == '1':
	            tmp_1 += 1
	        else:
	            tmp_0 += 1
	        for i, c in enumerate(s[1:]):
	            if p == c:
	                if p == '1':
	                    tmp_1 += 1
	                elif p == '0':
	                    tmp_0 += 1
	            else:
	                p = c
	                mx_1 = max(mx_1, tmp_1)
	                mx_0 = max(mx_0, tmp_0)
	                if c == '1':
	                    tmp_1 = 1
	                else:
	                    tmp_0 = 1
	        return max(mx_1, tmp_1) > max(mx_0, tmp_0)

### 运行结果

	Runtime: 14 ms, faster than 92.39% of Python online submissions for Longer Contiguous Segments of Ones than Zeros.
	Memory Usage: 13.3 MB, less than 90.22% of Python online submissions for Longer Contiguous Segments of Ones than Zeros.
	
原题链接：https://leetcode.com/problems/longer-contiguous-segments-of-ones-than-zeros



您的支持是我最大的动力
