leetcode  1910. Remove All Occurrences of a Substring（python）

### 描述


Given two strings s and part, perform the following operation on s until all occurrences of the substring part are removed:

* Find the leftmost occurrence of the substring part and remove it from s.
Return s after removing all occurrences of part.

A substring is a contiguous sequence of characters in a string.




Example 1:


	Input: s = "daabcbaabcbc", part = "abc"
	Output: "dab"
	Explanation: The following operations are done:
	- s = "daabcbaabcbc", remove "abc" starting at index 2, so s = "dabaabcbc".
	- s = "dabaabcbc", remove "abc" starting at index 4, so s = "dababc".
	- s = "dababc", remove "abc" starting at index 3, so s = "dab".
	Now s has no occurrences of "abc".
	
Example 2:

	Input: s = "axxxxyyyyb", part = "xy"
	Output: "ab"
	Explanation: The following operations are done:
	- s = "axxxxyyyyb", remove "xy" starting at index 4 so s = "axxxyyyb".
	- s = "axxxyyyb", remove "xy" starting at index 3 so s = "axxyyb".
	- s = "axxyyb", remove "xy" starting at index 2 so s = "axyb".
	- s = "axyb", remove "xy" starting at index 1 so s = "ab".
	Now s has no occurrences of "xy".




Note:
	
	1 <= s.length <= 1000
	1 <= part.length <= 1000
	s​​​​​​ and part consists of lowercase English letters.


### 解析

根据题意，给出一个字符串 s 和一个字符串 part ，对 s 执行以下操作，直到删除所有出现的子字符串 part：

* 找到最左边出现的子字符串 part 并将其从 s 中删除

删除所有出现的 part 后返回 s。同时题目还好心给出了子字符串的含义：子字符串是字符串中连续的字符序列。

其实看完题目之后我们就知道了，这个题很简单，就是考察字符串中的对字符的查找和索引的基本操作，使用一个 while 循环，使用 python 的内置函数 find 如果在 s 中能找到最左边 part 出现的起始索引 i 就让，就让 s[:i] + s[i+len(part):] 替换旧的 s ，直到 while 循环无法从 s 中找到 part 为止，最后返回 s 即可。


### 解答
				
	class Solution(object):
	    def removeOccurrences(self, s, part):
	        """
	        :type s: str
	        :type part: str
	        :rtype: str
	        """
	        while s.find(part)!=-1:
	            i = s.find(part)
	            s = s[:i] + s[i+len(part):]
	        return s
	            
            

            	      
			
### 运行结果

	Runtime: 16 ms, faster than 94.40% of Python online submissions for Remove All Occurrences of a Substring.
	Memory Usage: 13.6 MB, less than 61.60% of Python online submissions for Remove All Occurrences of a Substring.


原题链接：https://leetcode.com/problems/remove-all-occurrences-of-a-substring/



您的支持是我最大的动力
