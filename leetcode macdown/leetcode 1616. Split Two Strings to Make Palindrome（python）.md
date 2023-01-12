leetcode 1616. Split Two Strings to Make Palindrome （python）

### 描述

You are given two strings a and b of the same length. Choose an index and split both strings at the same index, splitting a into two strings: a<sub>prefix</sub> and a<sub>suffix</sub> where a = a<sub>prefix</sub> + a<sub>suffix</sub>, and splitting b into two strings: b<sub>prefix</sub> and b<sub>suffix</sub> where b = b<sub>prefix</sub> + b<sub>suffix</sub>. Check if a<sub>prefix</sub> + b<sub>suffix</sub> or b<sub>prefix</sub> + a<sub>suffix</sub> forms a palindrome.

When you split a string s into s<sub>prefix</sub> and s<sub>suffix</sub>, either s<sub>suffix</sub> or s<sub>prefix</sub> is allowed to be empty. For example, if s = "abc", then "" + "abc", "a" + "bc", "ab" + "c" , and "abc" + "" are valid splits.

Return true if it is possible to form a palindrome string, otherwise return false.

Notice that x + y denotes the concatenation of strings x and y.





Example 1:

	
* Input: a = "x", b = "y"
* Output: true
* Explaination: If either a or b are palindromes the answer is true since you can split in the following way:
* a<sub>prefix</sub> = "", a<sub>suffix</sub> = "x"
* b<sub>prefix</sub> = "", b<sub>suffix</sub> = "y"
* Then, a<sub>prefix</sub> + b<sub>suffix</sub> = "" + "y" = "y", which is a palindrome.
	
Example 2:

* Input: a = "abdef", b = "fecab"
* Output: true


Example 3:

* Input: a = "ulacfd", b = "jizalu"
* Output: true
* Explaination: Split them at index 3:
* a<sub>prefix</sub> = "ula", a<sub>suffix</sub> = "cfd"
* b<sub>prefix</sub> = "jiz", b<sub>suffix</sub> = "alu"
* Then, a<sub>prefix</sub> + b<sub>suffix</sub> = "ula" + "alu" = "ulaalu", which is a palindrome.

	
Example 4:


* Input: a = "xbdef", b = "xecab"
* Output: false
	


Note:

	1 <= a.length, b.length <= 10^5
	a.length == b.length
	a and b consist of lowercase English letters


### 解析

根据题意，给定两个长度相同的字符串 a 和 b。 选择一个索引并在同一索引处拆分两个字符串，将 a 拆分为两个字符串：a<sub>prefix</sub> 和 a<sub>suffix</sub>，其中 a = a<sub>prefix</sub> + a<sub>suffix</sub>，将 b 拆分为两个字符串：b<sub>prefix</sub> 和 b<sub>suffix</sub>，其中 b = b<sub>prefix</sub> + b<sub>suffix</sub>。 检查 a<sub>prefix</sub> + b<sub>suffix</sub> 或 b<sub>prefix</sub> + a<sub>suffix</sub> 是否形成回文。

将字符串 s 拆分为 s<sub>prefix</sub> 和 s<sub>suffix</sub> 时，允许 s<sub>suffix</sub> 或 s<sub>prefix</sub> 为空。 例如，如果 s = "abc"，则 "" + "abc"、"a" + "bc"、"ab" + "c" 和 "abc" + "" 是有效的拆分。如果可以形成回文字符串则返回 True ，否则返回 False。请注意，x + y 表示字符串 x 和 y 的串联。


其实这道题使用贪心的思想还是比较容易的，加入我们有两个字符串 a 和 b ，并且用中线假定划分的位置，如下：

* a：AB | CD | EF
* b：GH | KJ  | LM

如果 a 的前缀和 b 的后缀能组成回文，那么 AB 一定和 LM 能组成回文，那么只需要判断并且 CD 或者 KJ 是否是回文即可，即：AB+CD+LM 或者 AB+KJ+LM 这两种情况可以组成回文。将 b 的前缀和 a 的后缀组成回文也是相同的逻辑。



### 解答
				
	class Solution(object):
	    def checkPalindromeFormation(self, a, b):
	        """
	        :type a: str
	        :type b: str
	        :rtype: bool
	        """
	        return self.check(a, b) or self.check(b,a)
	    
	    def check(self, a, b):
	        if len(a) == 1 : return True
	        i = 0
	        j = len(a) - 1
	        while i<=j and a[i]==b[j]:
	            i+=1
	            j-=1
	        if i>j:return True
	        return self.isPalindrome(a[i:j+1]) or self.isPalindrome(b[i:j+1])
	    
	    def isPalindrome(self, s):
	        i = 0
	        j = len(s)-1
	        while i<=j and s[i]==s[j]:
	            i+=1
	            j-=1
	        return i>j
	        

            	      
			
### 运行结果

	Runtime: 119 ms, faster than 66.67% of Python online submissions for Split Two Strings to Make Palindrome.
	Memory Usage: 15.6 MB, less than 38.10% of Python online submissions for Split Two Strings to Make Palindrome.


原题链接：https://leetcode.com/problems/split-two-strings-to-make-palindrome/



您的支持是我最大的动力
