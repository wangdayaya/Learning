leetcode  1961. Check If String Is a Prefix of Array（python）

### 描述

Given a string s and an array of strings words, determine whether s is a prefix string of words.

A string s is a prefix string of words if s can be made by concatenating the first k strings in words for some positive k no larger than words.length.

Return true if s is a prefix string of words, or false otherwise.

 



Example 1:

	Input: s = "iloveleetcode", words = ["i","love","leetcode","apples"]
	Output: true
	Explanation:
	s can be made by concatenating "i", "love", and "leetcode" together.

	
Example 2:

	Input: s = "iloveleetcode", words = ["apples","i","love","leetcode"]
	Output: false
	Explanation:
	It is impossible to make s using a prefix of arr.




Note:

	1 <= words.length <= 100
	1 <= words[i].length <= 20
	1 <= s.length <= 1000
	words[i] and s consist of only lowercase English letters.



### 解析

根据题意，给出了一个字符串 s ，还有一个单词列表 words ，让我们判断 s 是否是 words 的前缀字符串。题目中给出了定义判断字符串 s 是否是列表 words 的前缀，只需要判断 s 是否和 words 中的前 k 个字符串拼接起来的字符串相同即可，k 要小于 words 的长度。

其实思路已经出来了：

* 初始化一个需要不断拼接的字符串 prefix
* 遍历 words 中的单词，将当前的单词 word 拼接到 prefix 后面，判断此时的 prefix 如果和 s 相等，直接返回 True ，否则重复此过程
* 遍历结束如果没有返回 True ，则表示没有找到 prefix ，直接返回 False


### 解答
				

	class Solution(object):
	    def isPrefixString(self, s, words):
	        """
	        :type s: str
	        :type words: List[str]
	        :rtype: bool
	        """
	        prefix = ''
	        for word in words:
	            prefix += word
	            if s==prefix:
	                return True
	        return False
            	      
			
### 运行结果

	Runtime: 20 ms, faster than 93.69% of Python online submissions for Check If String Is a Prefix of Array.
	Memory Usage: 13.4 MB, less than 71.17% of Python online submissions for Check If String Is a Prefix of Array.

### 解析

另外，我们可以直接将 words 中的单词拼接成一个字符串 r ，然后判断 s 是否是 r 的前缀。但是要保证不能是“伪前缀”，如 s=a ，words=["aa","aaaa","banana"] ，这种输入应该是 False ，所以还要保证 s 的长度必须是合理的。原理一样，换汤不换药。

### 解答
				
	 class Solution(object):
	    def isPrefixString(self, s, words):
	        """
	        :type s: str
	        :type words: List[str]
	        :rtype: bool
	        """
	        lengths = [len(words[0])]
	        for word in words[1:]:
	            lengths.append(len(word)+lengths[-1])
	        return len(s) in lengths and ''.join(words).startswith(s)           	      
			
### 运行结果
	Runtime: 20 ms, faster than 93.69% of Python online submissions for Check If String Is a Prefix of Array.
	Memory Usage: 13.6 MB, less than 22.97% of Python online submissions for Check If String Is a Prefix of Array.



原题链接：https://leetcode.com/problems/check-if-string-is-a-prefix-of-array/



您的支持是我最大的动力
