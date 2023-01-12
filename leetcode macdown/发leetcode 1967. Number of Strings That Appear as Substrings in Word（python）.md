leetcode  1967. Number of Strings That Appear as Substrings in Word（python）

### 描述

Given an array of strings patterns and a string word, return the number of strings in patterns that exist as a substring in word.

A substring is a contiguous sequence of characters within a string.





Example 1:

	Input: patterns = ["a","abc","bc","d"], word = "abc"
	Output: 3
	Explanation:
	- "a" appears as a substring in "abc".
	- "abc" appears as a substring in "abc".
	- "bc" appears as a substring in "abc".
	- "d" does not appear as a substring in "abc".
	3 of the strings in patterns appear as a substring in word.

	
Example 2:

	Input: patterns = ["a","b","c"], word = "aaaaabbbbb"
	Output: 2
	Explanation:
	- "a" appears as a substring in "aaaaabbbbb".
	- "b" appears as a substring in "aaaaabbbbb".
	- "c" does not appear as a substring in "aaaaabbbbb".
	2 of the strings in patterns appear as a substring in word.


Example 3:


	Input: patterns = ["a","a","a"], word = "ab"
	Output: 3
	Explanation: Each of the patterns appears as a substring in word "ab".
	


Note:

	1 <= patterns.length <= 100
	1 <= patterns[i].length <= 100
	1 <= word.length <= 100
	patterns[i] and word consist of lowercase English letters.


### 解析

根据题意，就是给出了一个列表 patterns ，里面有很多模版元素，又给出了一个单词 word ，要求我们找出有多少个模版在 word 中出现过，这个思路很简单，就是找子字符串的个数。直接遍历 patterns 中的每个元素，然后判断是否在 word 中出现过，将出现过的 pattern 进行计数，最后返回即可。


### 解答
				

	class Solution(object):
	    def numOfStrings(self, patterns, word):
	        """
	        :type patterns: List[str]
	        :type word: str
	        :rtype: int
	        """
	        result = 0
	        for p in patterns:
	            if p in word:
	                result += 1
	        return result
	                
            	      
			
### 运行结果

	
	Runtime: 20 ms, faster than 81.60% of Python online submissions for Number of Strings That Appear as Substrings in Word.
	Memory Usage: 13.7 MB, less than 18.40% of Python online submissions for Number of Strings That Appear as Substrings in Word.
	
### 解析


其实直接用 python 的内置函数 sum 也比较方便，一行代码就解决了，这真的是人生苦短我用 python 。

sum 函数本来就是对序列进行求和计算，例如：

	>>>sum([1,2,3])  
	6

	
如果对布尔类型的列表用此函数，会统计为 True 的个数，例如：
	
	>>>sum([True,True,False])  
	2
	
本题就是用到了 sum 函数的第二种计算方法，用来直接对 pattern in word 这个布尔结果进行计数，得到的结果即为答案。


### 解答
				

	class Solution(object):
	    def numOfStrings(self, patterns, word):
	        """
	        :type patterns: List[str]
	        :type word: str
	        :rtype: int
	        """
	        return sum(pattern in word for pattern in patterns)
	                
            	      
			
### 运行结果

	Runtime: 20 ms, faster than 81.60% of Python online submissions for Number of Strings That Appear as Substrings in Word.
	Memory Usage: 13.5 MB, less than 91.59% of Python online submissions for Number of Strings That Appear as Substrings in Word.
	
原题链接：https://leetcode.com/problems/number-of-strings-that-appear-as-substrings-in-word/



您的支持是我最大的动力
