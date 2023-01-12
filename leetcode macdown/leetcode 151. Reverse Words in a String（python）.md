leetcode 151. Reverse Words in a String （python）

### 描述


Given an input string s, reverse the order of the words.

A word is defined as a sequence of non-space characters. The words in s will be separated by at least one space.

Return a string of the words in reverse order concatenated by a single space.

Note that s may contain leading or trailing spaces or multiple spaces between two words. The returned string should only have a single space separating the words. Do not include any extra spaces.

Follow-up: If the string data type is mutable in your language, can you solve it in-place with O(1) extra space?


Example 1:

	Input: s = "the sky is blue"
	Output: "blue is sky the"

	
Example 2:


	Input: s = "  hello world  "
	Output: "world hello"
	Explanation: Your reversed string should not contain leading or trailing spaces.

Example 3:


	Input: s = "a good   example"
	Output: "example good a"
	Explanation: You need to reduce multiple spaces between two words to a single space in the reversed string.
	
Example 4:

	Input: s = "  Bob    Loves  Alice   "
	Output: "Alice Loves Bob"

	
Example 5:

	Input: s = "Alice does not even like bob"
	Output: "bob like even not does Alice"


Note:

	1 <= s.length <= 10^4
	s contains English letters (upper-case and lower-case), digits, and spaces ' '.
	There is at least one word in s.


### 解析


根据题意，给定一个输入字符串 s，反转单词的顺序。一个词被定义为一个非空格字符序列。 s 中的单词将被至少一个空格分隔。返回由单个空格连接的逆序单词字符串。请注意， s 可能包含两个单词之间的前导或尾随空格或多个空格。 返回的字符串中的单词应该只有一个空格分隔单词。题目给我们提出了要求，如果字符串是可变的，让我们用 O(1) 的空间复杂度。

不好意思，我的 python 字符串是不可变类型的，我只能用 O(n) 的空间复杂度，简单思路就是将 s 两边的空格先滤掉，然后将里面包含的单词切分出来，最后逆序遍历单词用空格拼接起来，这个思路是很简单易懂的。

### 解答
				

	class Solution(object):
	    def reverseWords(self, s):
	        """
	        :type s: str
	        :rtype: str
	        """
	        result = ''
	        s = s.strip()
	        for word in s.split()[::-1]:
	            result += word + ' '
	        return result.strip()
            	      
			
### 运行结果

	Runtime: 16 ms, faster than 93.13% of Python online submissions for Reverse Words in a String.
	Memory Usage: 13.8 MB, less than 67.19% of Python online submissions for Reverse Words in a String.


### 解析



### 解答



### 运行结果

原题链接：https://leetcode.com/problems/reverse-words-in-a-string/



您的支持是我最大的动力
