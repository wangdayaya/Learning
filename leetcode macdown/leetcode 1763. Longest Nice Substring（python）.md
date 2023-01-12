leetcode  1763. Longest Nice Substring（python）

### 描述


A string s is nice if, for every letter of the alphabet that s contains, it appears both in uppercase and lowercase. For example, "abABB" is nice because 'A' and 'a' appear, and 'B' and 'b' appear. However, "abA" is not because 'b' appears, but 'B' does not.

Given a string s, return the longest substring of s that is nice. If there are multiple, return the substring of the earliest occurrence. If there are none, return an empty string.

 


Example 1:

	Input: s = "YazaAay"
	Output: "aAa"
	Explanation: "aAa" is a nice string because 'A/a' is the only letter of the alphabet in s, and both 'A' and 'a' appear.
	"aAa" is the longest nice substring.

	
Example 2:

	Input: s = "Bb"
	Output: "Bb"
	Explanation: "Bb" is a nice string because both 'B' and 'b' appear. The whole string is a substring.


Example 3:


	Input: s = "c"
	Output: ""
	Explanation: There are no nice substrings.
	
Example 4:


	Input: s = "dDzeE"
	Output: "dD"
	Explanation: Both "dD" and "eE" are the longest nice substrings.
	As there are multiple longest nice substrings, return "dD" since it occurs earlier.



Note:
	
	1 <= s.length <= 100
	s consists of uppercase and lowercase English letters.


### 解析

根据题意，就是找出一个最长的 nice 字符串，这种字符串是 s 的子字符串且其中必然同时包含了某个英文字母的大小写。使用暴力法破解，双重循环遍历所有可能的子字符串，判断是否是 nice ，如果是并且长度较大，则赋值给 result ，遍历结束得到的 result 即为结果。


### 解答
				

	class Solution(object):
	    def longestNiceSubstring(self, s):
	        """
	        :type s: str
	        :rtype: str
	        """
	        def isNice(string):
	            string_ = string.swapcase()
	            string_list = list(string_)
	            stringlist = list(string)
	            for c in stringlist:
	                if c not in string_list:
	                    return False
	            return True
	        result = ''
	        for i in range(len(s)-1):
	            for j in range(i+1,len(s)):
	                tmp = s[i:j+1]
	                if isNice(tmp) and len(tmp)>len(result):
	                    result = tmp
	        return result
            	      
			
### 运行结果
	Runtime: 280 ms, faster than 20.59% of Python online submissions for Longest Nice Substring.
	Memory Usage: 13.4 MB, less than 93.14% of Python online submissions for Longest Nice Substring.


### 解析

另外可以用递归的方法，在 s 中找到一些字符放入列表 p 中，这些字符的大/小写不在 s 中，说明其不具备成为 NiceSubstring ，所以以这些字符作为分割界限，划为左右两个字符串，然后递归分别为左/右字符串继续找下去，如果某个字符串中的 p 的长度为 0 ，说明其本身是 NiceSubstring ，直接返回即可，最后将找出的所有的 NiceSubstring 进行排序，即可找到出现最早的并且是最长的 NiceSubstring 。


### 解答

	class Solution(object):
	    def longestNiceSubstring(self, s):
	        """
	        :type s: str
	        :rtype: str
	        """
	        def discovery(s):
	            if len(s) < 2:
	                return ""
	            p = []
	            for i, ch in enumerate(s):
	                if ch.isupper() and ch.lower() not in s:
	                    p.append(i)
	                if ch.islower() and ch.upper() not in s:
	                    p.append(i)
	            if not p:
	                return s
	            else:
	                mid = (len(p)) // 2
	                left = s[:p[mid]]
	                right = s[p[mid]+1:]
	                return max(discovery(left),discovery(right), key=len)
	        return discovery(s)

### 运行结果

	Runtime: 32 ms, faster than 68.37% of Python online submissions for Longest Nice Substring.
	Memory Usage: 13.6 MB, less than 52.04% of Python online submissions for Longest Nice Substring.


原题链接：https://leetcode.com/problems/longest-nice-substring/



您的支持是我最大的动力
