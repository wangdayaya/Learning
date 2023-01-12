leetcode  2042. Check if Numbers Are Ascending in a Sentence（python）

### 描述

A sentence is a list of tokens separated by a single space with no leading or trailing spaces. Every token is either a positive number consisting of digits 0-9 with no leading zeros, or a word consisting of lowercase English letters.

* For example, "a puppy has 2 eyes 4 legs" is a sentence with seven tokens: "2" and "4" are numbers and the other tokens such as "puppy" are words.

Given a string s representing a sentence, you need to check if all the numbers in s are strictly increasing from left to right (i.e., other than the last number, each number is strictly smaller than the number on its right in s).

Return true if so, or false otherwise.



Example 1:

![](https://assets.leetcode.com/uploads/2021/09/30/example1.png)

	Input: s = "1 box has 3 blue 4 red 6 green and 12 yellow marbles"
	Output: true
	Explanation: The numbers in s are: 1, 3, 4, 6, 12.
	They are strictly increasing from left to right: 1 < 3 < 4 < 6 < 12.

	
Example 2:


	Input: s = "hello world 5 x 5"
	Output: false
	Explanation: The numbers in s are: 5, 5. They are not strictly increasing.

Example 3:

![](https://assets.leetcode.com/uploads/2021/09/30/example3.png)

	Input: s = "sunset is at 7 51 pm overnight lows will be in the low 50 and 60 s"
	Output: false
	Explanation: The numbers in s are: 7, 51, 50, 60. They are not strictly increasing.

	
Example 4:

	Input: s = "4 5 11 26"
	Output: true
	Explanation: The numbers in s are: 4, 5, 11, 26.
	They are strictly increasing from left to right: 4 < 5 < 11 < 26.




Note:

	3 <= s.length <= 200
	s consists of lowercase English letters, spaces, and digits from 0 to 9, inclusive.
	The number of tokens in s is between 2 and 100, inclusive.
	The tokens in s are separated by a single space.
	There are at least two numbers in s.
	Each number in s is a positive number less than 100, with no leading zeros.
	s contains no leading or trailing spaces.


### 解析


根据题意，给出了一个字符串句子 s ，s 中包含了用单个空格分开的正整数子字符串或者小写英文单词，并且 s 没有前置或者后置的空格。如 "a puppy has 2 eyes 4 legs" 就是符合题意的字符串。题目要求我们判断是否字符串 s 中的所有数字都是严格升序的。如果是返回 True ，否则返回 False 。

题目中给出了严格升序的概念：除了最后一个数字，每个数字都严格小于在 s 中它的右边的数字。

思路比较简单：

* 初始化上个最大数字 last 为 0 
* 使用内置函数 s.split(' ') 先将 s 拆分成子字符串列表 L
* 然后遍历列表 L ，如果元素 c 是数字字符串，那么如果 int(c) 大于 last ，则更新 last 为 int(c) ，否则直接返回 False 
* 遍历结束，数字都满足严格升序，直接返回 True

### 解答
				

	class Solution(object):
	    def areNumbersAscending(self, s):
	        """
	        :type s: str
	        :rtype: bool
	        """
	        L = s.split(' ')
	        numbers = '1234567890'
	        last = 0
	        for c in L:
	            if c[0] in numbers:
	                if int(c)>last:
	                    last = int(c)
	                else:
	                    return False
	        return True
            	      
			
### 运行结果


	Runtime: 20 ms, faster than 100.00% of Python online submissions for Check if Numbers Are Ascending in a Sentence.
	Memory Usage: 13.6 MB, less than 50.00% of Python online submissions for Check if Numbers Are Ascending in a Sentence.


### 解析

内置函数也可以快速解题，直接使用正则函数找到所有的数字，然后判断数字列表是否是升序即可。

### 解答

	
	class Solution(object):
	    def areNumbersAscending(self, s):
	        """
	        :type s: str
	        :rtype: bool
	        """
	        nums = re.findall(r'\d+', s)
	        return nums == sorted(set(nums), key=int)

### 运行结果

	Runtime: 24 ms, faster than 100.00% of Python online submissions for Check if Numbers Are Ascending in a Sentence.
	Memory Usage: 13.4 MB, less than 50.00% of Python online submissions for Check if Numbers Are Ascending in a Sentence.

原题链接：https://leetcode.com/problems/check-if-numbers-are-ascending-in-a-sentence/



您的支持是我最大的动力
