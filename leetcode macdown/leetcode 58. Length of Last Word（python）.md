leetcode  58. Length of Last Word（python）

### 描述


Given a string s consisting of some words separated by some number of spaces, return the length of the last word in the string.

A word is a maximal substring consisting of non-space characters only.




Example 1:


	Input: s = "Hello World"
	Output: 5
	Explanation: The last word is "World" with length 5.
	
Example 2:

	Input: s = "   fly me   to   the moon  "
	Output: 4
	Explanation: The last word is "moon" with length 4.


Example 3:


	Input: s = "luffy is still joyboy"
	Output: 6
	Explanation: The last word is "joyboy" with length 6.
	




Note:


	1 <= s.length <= 10^4
	s consists of only English letters and spaces ' '.
	There will be at least one word in s.

### 解析

根据题意，给出了一个字符串 s ，是由若干个小写英文单词和空格组成的，要求我们返回最后一个小写英文单词的长度。其实这里的单词也不是严格意义上的单词，只是由非空格组成的子字符串。
思路比较简单，这道题就是考察对字符串的空格的处理和对字符串的分割，我直接使用 python 内置函数 strip 先去掉 s 两端的空格，然后用 split 函数通过空格分割 s 得到子字符串列表，列表最后一个元素肯定是单词，求其长度即可。

当然也可以不用内置函数，通过遍历字符串 s 也可以进行求解，总之这道题比较简单，我用内置函数也是为了快速刷题。
### 解答
				
	class Solution(object):
	    def lengthOfLastWord(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        return len(s.strip().split(' ')[-1])

            	      
			
### 运行结果
	
	Runtime: 32 ms, faster than 12.29% of Python online submissions for Length of Last Word.
	Memory Usage: 13.7 MB, less than 55.34% of Python online submissions for Length of Last Word.


原题链接：https://leetcode.com/problems/length-of-last-word/



您的支持是我最大的动力
