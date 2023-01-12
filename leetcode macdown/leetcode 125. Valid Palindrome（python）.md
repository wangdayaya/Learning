leetcode  125. Valid Palindrome（python）

### 描述

Given a string s, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.





Example 1:

	Input: s = "A man, a plan, a canal: Panama"
	Output: true
	Explanation: "amanaplanacanalpanama" is a palindrome.

	
Example 2:

	Input: s = "race a car"
	Output: false
	Explanation: "raceacar" is not a palindrome.





Note:

	1 <= s.length <= 2 * 10^5
	s consists only of printable ASCII characters.


### 解析


根据题意，就是给出了一个字符串 s ，s 中包含了数字或者大小写字母或者空格，题目要求我们判断这个字符串 s 中在忽略大小写、空格等的基础上，是不是一个回文字符串。回文字符串的特征就是以中心线镜面对称的字符串。再说的简单点就是一个正读和反读都一样的字符串。思路比较简单：

* 初始化两个指针 i 和 j ，i 为 0 指向字符串开头，j 为 len(s)-1 指向字符串的结尾
* 当 i<=j 的时候进行 while 循环
* 先找到第一个正向的合法字符索引 i ，再找到第一个逆向的合法字符索引 j ，判断两者在小写状态下是否相等，如果不等直接返回 False ，如果相等则 i 加一，j 减一
* 当 while 条件允许的情况下，再找到正向第二个合法字符索引 i ，逆向第二个合法字符索引 j ，进行和上述相同的操作，判断两个字符的小写状态下是否相等，如果不等直接返回 False ，如果相等则 i 加一，j 减一
* 一直循环下去，直到最后 j<i 跳出 while 循环
* 循环结束返回 True


### 解答
				
	class Solution(object):
	    def isPalindrome(self, s):
	        """
	        :type s: str
	        :rtype: bool
	        """
	        i = 0
	        j = len(s)-1
	        while i<=j:
	            while not s[i].isalnum() and i<len(s)-1:
	                i += 1
	            while not s[j].isalnum() and j>=0:
	                j -= 1
	            if s[i].lower() != s[j].lower():
	                return False
	            i += 1
	            j -= 1
	        return True
	                
	                

            	      
			
### 运行结果

	Runtime: 44 ms, faster than 69.81% of Python online submissions for Valid Palindrome.
	Memory Usage: 14.2 MB, less than 70.02% of Python online submissions for Valid Palindrome.


原题链接：https://leetcode.com/problems/valid-palindrome/



您的支持是我最大的动力
