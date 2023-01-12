leetcode  1433. Check If a String Can Break Another String（python）

### 描述


Given two strings: s1 and s2 with the same size, check if some permutation of string s1 can break some permutation of string s2 or vice-versa. In other words s2 can break s1 or vice-versa.

A string x can break string y (both of size n) if x[i] >= y[i] (in alphabetical order) for all i between 0 and n-1.


Example 1:

	Input: s1 = "abc", s2 = "xya"
	Output: true
	Explanation: "ayx" is a permutation of s2="xya" which can break to string "abc" which is a permutation of s1="abc".

	
Example 2:

	Input: s1 = "abe", s2 = "acd"
	Output: false 
	Explanation: All permutations for s1="abe" are: "abe", "aeb", "bae", "bea", "eab" and "eba" and all permutation for s2="acd" are: "acd", "adc", "cad", "cda", "dac" and "dca". However, there is not any permutation from s1 which can break some permutation from s2 and vice-versa.


Example 3:


	Input: s1 = "leetcodee", s2 = "interview"
	Output: true



Note:

	s1.length == n
	s2.length == n
	1 <= n <= 10^5
	All strings consist of lowercase English letters.


### 解析


根据题意，给定长度相同的两个字符串 s1 和 s2，检查字符串 s1 的某些排列是否可以破坏字符串 s2 的某些排列，反之亦然。 简单说就是，s2 可以破坏 s1，反之亦然。如果对于 0 和 n-1 之间的所有 i ，x[i] >= y[i]（按字母顺序），则表示字符串 x 可以断开字符串 y（长度均为 n ）。

其实看完题目我们就知道其实就是判断 s1 比 s2 对应位置上的字符都大或者都小，这种情况就要都对 s1 和 s2 进行升序排序，然后比较是不是满足 s1 比 s2 对应位置上的字符都大或者都小。



### 解答
				
	class Solution(object):
	    def checkIfCanBreak(self, s1, s2):
	        """
	        :type s1: str
	        :type s2: str
	        :rtype: bool
	        """
	        s1 = list(s1)
	        s2 = list(s2)
	        s1.sort()
	        s2.sort()
	        def check(s1, s2):
	            for i in range(len(s1)):
	                if s1[i] < s2[i]:
	                    return False
	            return True
	        return check(s1, s2) or check(s2, s1)

            	      
			
### 运行结果


	Runtime: 172 ms, faster than 86.67% of Python online submissions for Check If a String Can Break Another String.
	Memory Usage: 19.9 MB, less than 53.33% of Python online submissions for Check If a String Can Break Another String.


### 解析

和上面的原理类似，只不过是判断 s1 比 s2 对应位置字符都大或者都小的个数是否等于 s1 的长度。

### 解答
	class Solution(object):
	    def checkIfCanBreak(self, s1, s2):
	        """
	        :type s1: str
	        :type s2: str
	        :rtype: bool
	        """
	        s1 = sorted(s1)
	        s2 = sorted(s2)
	        a = b = 0
	        n = len(s1)
	        for i in range(len(s1)):
	            if s1[i] <= s2[i]:
	                a += 1
	            if s1[i] >= s2[i]:
	                b += 1
	        return a == n or b == n
	
			
### 运行结果

	Runtime: 228 ms, faster than 53.33% of Python online submissions for Check If a String Can Break Another String.
	Memory Usage: 20.2 MB, less than 26.67% of Python online submissions for Check If a String Can Break Another String.


原题链接：https://leetcode.com/problems/check-if-a-string-can-break-another-string/



您的支持是我最大的动力
