leetcode 680. Valid Palindrome II （python）




### 描述


Given a string s, return true if the s can be palindrome after deleting at most one character from it.

 


Example 1:


	Input: s = "aba"
	Output: true
	
Example 2:


	Input: s = "abca"
	Output: true
	Explanation: You could delete the character 'c'.

Example 3:

	Input: s = "abc"
	Output: false

	




Note:


	1 <= s.length <= 10^5
	s consists of lowercase English letters.

### 解析

根据题意，给定一个字符串 s ，如果删除最多一个字符后 s 可以是回文字符串，则返回 true ，否则返回 false 。

其实这道题就是考察回文字符串概念，回文字符串说白了就是字符串是镜像对称的，第一个字符和最后一个字符相同，第二个和倒数第二个字符相同，第三个和倒数第三个字符相同。。。以此类推。这个概念就是我们判断字符串是否是回文字符串的核心方法，这里我定义了函数 check 来检查一个字符串是否是回文的。

如果经过 check 判断 s 本身就是回文的那就最好不过，直接返回 True ，关键是有可能 s 不是回文的，而且题目限制我们只允许删除最多一个字符，换句话说我们只允许一个字符的位置不能满足回文规律，所以我们可以这样做：

* 在合法的情况下，用两个索引  i 和 j 分别从两头向中间靠拢查询
* 当碰到 s[i]  和 s[j] 相等的情况下， 直接将 i 加一 ，j 减 1 
* 如果 s[i]  和 s[j] 不相等的情况下，我们就判断 (check(s[i:j]) or check(s[i+1:j+1]) 是否为 True ，如果为 False 则直接返回 False ，说明在去掉一个字符之后剩下的子字符串仍不构成回文字符串，如果为 True 则直接返回 True ，说明在去掉一个字符之后剩下的子字符串可能构成回文字符串

时间复杂度为 O(N) ，空间复杂度为 O(n) 。


### 解答
				
	class Solution(object):
	    def validPalindrome(self, s):
	        """
	        :type s: str
	        :rtype: bool
	        """
	        def check(s):
	            i = 0
	            j = len(s) -1
	            while i<j:
	                if s[i] == s[j]:
	                    i += 1
	                    j -= 1
	                else:
	                    return False
	            return True
	        if check(s) : return True
	        i = 0
	        j = len(s) -1
	        while i<j:
	            if s[i] == s[j]:
	                i += 1
	                j -= 1
	            else:
	                return check(s[i:j]) or check(s[i+1:j+1])
	        return True
	                    
	        
### 运行结果

	Runtime: 148 ms, faster than 59.99% of Python online submissions for Valid Palindrome II.
	Memory Usage: 14.1 MB, less than 62.72% of Python online submissions for Valid Palindrome II.




### 原题链接

https://leetcode.com/problems/valid-palindrome-ii/


您的支持是我最大的动力
