leetcode  2315. Count Asterisks（python）




### 描述

You are given a string s, where every two consecutive vertical bars '|' are grouped into a pair. In other words, the 1st and 2nd '|' make a pair, the 3rd and 4th '|' make a pair, and so forth.

Return the number of '\*' in s, excluding the '\*' between each pair of '|'.

Note that each '|' will belong to exactly one pair.



Example 1:

	Input: s = "l|*e*et|c**o|*de|"
	Output: 2
	Explanation: The considered characters are underlined: "l|*e*et|c**o|*de|".
	The characters between the first and second '|' are excluded from the answer.
	Also, the characters between the third and fourth '|' are excluded from the answer.
	There are 2 asterisks considered. Therefore, we return 2.

	
Example 2:

	Input: s = "iamprogrammer"
	Output: 0
	Explanation: In this example, there are no asterisks in s. Therefore, we return 0.


Example 3:


	Input: s = "yo|uar|e**|b|e***au|tifu|l"
	Output: 5
	Explanation: The considered characters are underlined: "yo|uar|e**|b|e***au|tifu|l". There are 5 asterisks considered. Therefore, we return 5.
	



Note:

	1 <= s.length <= 1000
	s consists of lowercase English letters, vertical bars '|', and asterisks '*'.
	s contains an even number of vertical bars '|'.


### 解析

根据题意，给定一个字符串 s ，其中每两个连续的竖线 '|'  中包含的字符串是合法的字符串。 换句话说，第一个和第二个 '|' 中的字符串是合法的，第三个和第四个 '|' 中的字符串是合法的，依此类推。返回 s 中不合法的字符串中的 '\*' 的个数。

这道题其实很简单，只不过英文题目描述起来不那么容易理解，我们只需要换个思路，不去被这个 “｜” 所束缚，反正目标是找不合法字符串中的星号数量，我们只需要对 s 按照 “｜” 进行分割成字符串列表 L ，此时偶数索引的字符串就是题目中说的不合法字符串，然后只需要找到偶数索引的字符串，将他们里面的星号数量进行计数然后加到 result 中即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				
	class Solution(object):
	    def countAsterisks(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        result = 0
	        L = s.split('|')
	        N = len(L)
	        for i in range(0,N,2):
	            result += L[i].count('*')
	        return result
   
			
### 运行结果

	
	69 / 69 test cases passed.
	Status: Accepted
	Runtime: 41 ms
	Memory Usage: 13.6 MB


### 原题链接

https://leetcode.com/contest/biweekly-contest-81/problems/count-asterisks/


您的支持是我最大的动力
