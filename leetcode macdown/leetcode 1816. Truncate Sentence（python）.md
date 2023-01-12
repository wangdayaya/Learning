leetcode  1816. Truncate Sentence（python）

### 描述


A sentence is a list of words that are separated by a single space with no leading or trailing spaces. Each of the words consists of only uppercase and lowercase English letters (no punctuation).

* For example, "Hello World", "HELLO", and "hello world hello world" are all sentences.

You are given a sentence s​​​​​​ and an integer k​​​​​​. You want to truncate s​​​​​​ such that it contains only the first k​​​​​​ words. Return s​​​​​​ after truncating it.

 


Example 1:


	Input: s = "Hello how are you Contestant", k = 4
	Output: "Hello how are you"
	Explanation:
	The words in s are ["Hello", "how" "are", "you", "Contestant"].
	The first 4 words are ["Hello", "how", "are", "you"].
	Hence, you should return "Hello how are you".
	
Example 2:

	Input: s = "What is the solution to this problem", k = 4
	Output: "What is the solution"
	Explanation:
	The words in s are ["What", "is" "the", "solution", "to", "this", "problem"].
	The first 4 words are ["What", "is", "the", "solution"].
	Hence, you should return "What is the solution".


Example 3:


	Input: s = "chopper is not a tanuki", k = 5
	Output: "chopper is not a tanuki"
	



Note:

	1 <= s.length <= 500
	k is in the range [1, the number of words in s].
	s consist of only lowercase and uppercase English letters and spaces.
	The words in s are separated by a single space.
	There are no leading or trailing spaces.



### 解析


根据题意，就是将句子中的前 k 个单词用空格拼接起来，用内置函数也就两行代码，我这里自己直接遍历字符串，通过判断拼接得到最后的结果，这或许就是自己给自己增加难度吧。

### 解答
				

	class Solution(object):
	    def truncateSentence(self, s, k):
	        """
	        :type s: str
	        :type k: int
	        :rtype: str
	        """
	        r = ''
	        index = 0
	        while k and index<len(s):
	            if s[index]==' ':
	                if index==0:
	                    index += 1
	                elif index>=1 and s[index-1]==' ':
	                    index += 1
	                elif index>=1 and s[index-1]!=' ':
	                    index += 1
	                    k -= 1
	                    if k>=1:
	                        r += ' '
	            elif s[index]!=' ':
	                r += s[index]
	                index += 1
	        return r
            	      
			
### 运行结果

	Runtime: 16 ms, faster than 80.21% of Python online submissions for Truncate Sentence.
	Memory Usage: 13.6 MB, less than 41.33% of Python online submissions for Truncate Sentence.


### 解析

使用内置函数



### 解答
				

	class Solution(object):
	    def truncateSentence(self, s, k):
	        """
	        :type s: str
	        :type k: int
	        :rtype: str
	        """
	        words = s.split(" ")
	        return " ".join(words[0:k])
            	      
			
### 运行结果

	Runtime: 12 ms, faster than 96.92% of Python online submissions for Truncate Sentence.
	Memory Usage: 13.3 MB, less than 93.33% of Python online submissions for Truncate Sentence.



原题链接：https://leetcode.com/problems/truncate-sentence/



您的支持是我最大的动力
