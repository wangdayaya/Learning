leetcode  1328. Break a Palindrome（python）




### 描述

Given a palindromic string of lowercase English letters palindrome, replace exactly one character with any lowercase English letter so that the resulting string is not a palindrome and that it is the lexicographically smallest one possible. Return the resulting string. If there is no way to replace a character to make it not a palindrome, return an empty string.

A string a is lexicographically smaller than a string b (of the same length) if in the first position where a and b differ, a has a character strictly smaller than the corresponding character in b. For example, "abcc" is lexicographically smaller than "abcd" because the first position they differ is at the fourth character, and 'c' is smaller than 'd'.



Example 1:

	Input: palindrome = "abccba"
	Output: "aaccba"
	Explanation: There are many ways to make "abccba" not a palindrome, such as "zbccba", "aaccba", and "abacba".
	Of all the ways, "aaccba" is the lexicographically smallest.

	
Example 2:

	Input: palindrome = "a"
	Output: ""
	Explanation: There is no way to replace a single character to make "a" not a palindrome, so return an empty string.




Note:

	1 <= palindrome.length <= 1000
	palindrome consists of only lowercase English letters.


### 解析

根据题意，给定一个由小写英语字母组成的回文字符串 palindrome ，将一个字符替换为任何小写英语字母，以便生成的字符串不是回文，并且它是词典序最小的一个可能的回文结果。返回生成的字符串，如果无法替换字符，则返回空字符串。

长度相同的字符串 a 和 b ，如果字符串 a 的第一个字符在字典序上小于字符串 b 的第一个字符，那么就说 a 是字典序严格小于 b ，如果第一个字符相同，按照相同的规则查比较第二个字符、第三个字符，以此类推。例如，“abcc”在字典序上小于“abcd”，因为它们的第四个字符处不相同，并且“c”小于“d”。

这种题就属于看着很容易，然后提交不断爆红，然后越想越复杂，但是最后看了题解发现又很简单的贪心类型题，说到底还是没有完全把不同的情况归纳清楚，对于这道题目想要打破字符串的会文规律，并且还要使得结果字符串的字典序最小，这个目标我们一定要记住，难点就在于回文字符串的长度以及回文字符串可能出现都是 a 的情况。

我们知道回文字符串有两种情况，一种是奇数长度，一种是偶数长度，对于奇数长度的回文字符串中间的字符不管怎么变化都还是回文字符串，偶数长度的回文字符串不存在这种情况。另外回文字符串前半个和后半个都是一一对应的，所以我们只需要遍历处理前面一半的字符串即可。遇到不是 a 的字符，直接将其替换成 a 即可满足题意将其直接返回即可。但是如果前半个字符串都是 a ，那么说明后半个字符串也都是 a ，那直接将字符串的最后一个字符变成 b 即可满足题目的条件，并将其返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。
 

### 解答

	class Solution(object):
	    def breakPalindrome(self, palindrome):
	        """
	        :type palindrome: str
	        :rtype: str
	        """
	        N = len(palindrome)
	        if N < 2:
	            return ""
	        for i in range(N >> 1):
	            if palindrome[i] > 'a':
	                return palindrome[:i] + 'a' + palindrome[i + 1:]
	        return palindrome[:N - 1] + 'b'

### 运行结果

	Runtime: 13 ms, faster than 96.92% of Python online submissions for Break a Palindrome.
	Memory Usage: 13.5 MB, less than 31.65% of Python online submissions for Break a Palindrome.

### 原题链接

https://leetcode.com/problems/break-a-palindrome/


您的支持是我最大的动力
