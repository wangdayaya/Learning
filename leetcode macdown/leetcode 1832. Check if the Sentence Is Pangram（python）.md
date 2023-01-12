leetcode  1832. Check if the Sentence Is Pangram（python）

### 描述

A pangram is a sentence where every letter of the English alphabet appears at least once.

Given a string sentence containing only lowercase English letters, return true if sentence is a pangram, or false otherwise.



Example 1:

	Input: sentence = "thequickbrownfoxjumpsoverthelazydog"
	Output: true
	Explanation: sentence contains at least one of every letter of the English alphabet.

	
Example 2:

	Input: sentence = "leetcode"
	Output: false






Note:

	1 <= sentence.length <= 1000
	sentence consists of lowercase English letters.


### 解析

根据题意，就是找出是不是一个字符串中包含了所有的小写英文字母。解法很简单，统计字符串中的字符的出现个数形成字典 c ，判断 c 中包含的键数量是否大于等于 26 即可。


### 解答
				
	class Solution(object):
	    def checkIfPangram(self, sentence):
	        """
	        :type sentence: str
	        :rtype: bool
	        """
	        if len(sentence)<26:
	            return False
	        return len(collections.Counter(sentence).keys())>=26

            	      
			
### 运行结果

	Runtime: 24 ms, faster than 36.28% of Python online submissions for Check if the Sentence Is Pangram.
	Memory Usage: 13.5 MB, less than 31.53% of Python online submissions for Check if the Sentence Is Pangram.


原题链接：https://leetcode.com/problems/check-if-the-sentence-is-pangram/



您的支持是我最大的动力
