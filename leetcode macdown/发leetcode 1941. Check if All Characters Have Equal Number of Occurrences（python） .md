leetcode  1941. Check if All Characters Have Equal Number of Occurrences（python）

### 描述


Given a string s, return true if s is a good string, or false otherwise.

A string s is good if all the characters that appear in s have the same number of occurrences (i.e., the same frequency).




Example 1:


	Input: s = "abacbc"
	Output: true
	Explanation: The characters that appear in s are 'a', 'b', and 'c'. All characters occur 2 times in s.
	
Example 2:

	Input: s = "aaabb"
	Output: false
	Explanation: The characters that appear in s are 'a' and 'b'.
	'a' occurs 3 times while 'b' occurs 2 times, which is not the same number of times.




Note:

	1 <= s.length <= 1000
	s consists of lowercase English letters.



### 解析

根据题意，就是给出一个字符串 s ，判断这个字符串 s 是不是 good 。题目中给出了 good 字符串的定义，就是字符串 s 中的每个出现的字符的个数相等。思路很简单：

* 初始化一个计数器 count
* 遍历 s 中的每个字符，对他们出现的数量进行计数
* 判断 count.values() 列表中的数字在去重之后长度是否为 1 ，如果为 1 则表示 s 是 good 字符串，否则不是 good 字符串


### 解答
					
	class Solution(object):
	    def areOccurrencesEqual(self, s):
	        """
	        :type s: str
	        :rtype: bool
	        """
	        count = {}
	        for c in s:
	            if c not in count:
	                count[c] = 1
	            else:
	                count[c] += 1
	        return len(set(count.values()))==1

            	      
			
### 运行结果


	Runtime: 24 ms, faster than 80.22% of Python online submissions for Check if All Characters Have Equal Number of Occurrences.
	Memory Usage: 13.6 MB, less than 13.55% of Python online submissions for Check if All Characters Have Equal Number of Occurrences.

### 解析


本质上这个题目考的就是计数器的使用，所以可以借用 python 的内置函数 collections.Counter() 得到对 s 中的字符的技术结果 c ，然后对 c.values() 去重判断是否长度为 1 ，和上面思路一样。

### 解答
					
	class Solution(object):
	    def areOccurrencesEqual(self, s):
	        """
	        :type s: str
	        :rtype: bool
	        """
	        c = collections.Counter(s)
	        return len(set(c.values()))==1
			
### 运行结果

	Runtime: 32 ms, faster than 38.10% of Python online submissions for Check if All Characters Have Equal Number of Occurrences.
	Memory Usage: 13.5 MB, less than 70.33% of Python online submissions for Check if All Characters Have Equal Number of Occurrences.
	
原题链接：https://leetcode.com/problems/check-if-all-characters-have-equal-number-of-occurrences/



您的支持是我最大的动力
