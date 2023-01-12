leetcode  2047. Number of Valid Words in a Sentence（python）

### 描述


A sentence consists of lowercase letters ('a' to 'z'), digits ('0' to '9'), hyphens ('-'), punctuation marks ('!', '.', and ','), and spaces (' ') only. Each sentence can be broken down into one or more tokens separated by one or more spaces ' '.

A token is a valid word if all three of the following are true:

* It only contains lowercase letters, hyphens, and/or punctuation (no digits).
* There is at most one hyphen '-'. If present, it must be surrounded by lowercase characters ("a-b" is valid, but "-ab" and "ab-" are not valid).
* There is at most one punctuation mark. If present, it must be at the end of the token ("ab,", "cd!", and "." are valid, but "a!b" and "c.," are not valid).

Examples of valid words include "a-b.", "afad", "ba-c", "a!", and "!".

Given a string sentence, return the number of valid words in sentence.


Example 1:


	Input: sentence = "cat and  dog"
	Output: 3
	Explanation: The valid words in the sentence are "cat", "and", and "dog".
	
Example 2:

	Input: sentence = "!this  1-s b8d!"
	Output: 0
	Explanation: There are no valid words in the sentence.
	"!this" is invalid because it starts with a punctuation mark.
	"1-s" and "b8d" are invalid because they contain digits.


Example 3:


	Input: sentence = "alice and  bob are playing stone-game10"
	Output: 5
	Explanation: The valid words in the sentence are "alice", "and", "bob", "are", and "playing".
	"stone-game10" is invalid because it contains digits.
	
Example 4:


	Input: sentence = "he bought 2 pencils, 3 erasers, and 1  pencil-sharpener."
	Output: 6
	Explanation: The valid words in the sentence are "he", "bought", "pencils,", "erasers,", "and", and "pencil-sharpener.".
	

Note:


	1 <= sentence.length <= 1000
	sentence only contains lowercase English letters, digits, ' ', '-', '!', '.', and ','.
	There will be at least 1 token.

### 解析


根据题意，给出一个字符串 sentence 由小写字母（'a' 到 'z'）、数字（'0' 到 '9'）、连字符（'-'）、标点符号（'!'、'.' 和 ','）组成和空格 (' ') 组成。 每个句子都可以通过为一个或多个空格分割为多个 token 。

如果以下所有三个条件都成立都为 True ，token 为有效的：

* 它仅包含小写字母、连字符，还可能有标点符号（无数字）。
* 最多有一个连字符“-”。 如果存在，它必须用小写字符包围（“a-b”有效，但“-ab”和“ab-”无效）。
* 最多有一个标点符号。 如果存在，它必须位于标记的末尾（“ab,”、“cd!”和“.”有效，但“a!b”和“c.,”无效）。

有效词的示例包括“a-b.”、“afad”、“ba-c”、“a!” 和 “!”。给定一个字符串 sentence ，返回句子中有效单词的数量。

其实题目比较繁杂， 但是解决方法很简单，就是判断题目中的三个要求即可：

* 将 sentence 分成多个 token
* 遍历每个 token ，如果里面包含数字直接进行下一个 token 的判断，如果里面包含的连字符 - 多于一个或者不在字母中间直接进行下一个 token 的判断，如果标点多于一个或者不在末尾直接进行下一个 token 的判断，否则计数器加一
* 遍历结束返回计数器结果

### 解答
				

	class Solution(object):
	    def countValidWords(self, sentence):
	        """
	        :type sentence: str
	        :rtype: int
	        """
	        result = 0
	        tokens = sentence.split()
	        for token in tokens:
	            if re.search(r'\d', token):
	                continue
	            if token.count('-')>1 or ('-' in token and not re.search(r'[a-z]-[a-z]', token)):
	                continue
	            count = token.count('!') + token.count('.') + token.count(',') 
	            if count > 1:
	                continue
	            if count == 1:
	                if token[-1] not in ['!', '.' , ',']:
	                    continue
	            result += 1
	        return result
	            
	                
	            
            	      
			
### 运行结果

	Runtime: 40 ms, faster than 75.10% of Python online submissions for Number of Valid Words in a Sentence.
	Memory Usage: 13.6 MB, less than 72.65% of Python online submissions for Number of Valid Words in a Sentence.


原题链接：https://leetcode.com/problems/number-of-valid-words-in-a-sentence/



您的支持是我最大的动力
