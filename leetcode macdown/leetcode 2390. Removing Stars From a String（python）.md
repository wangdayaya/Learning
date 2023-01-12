leetcode  2390. Removing Stars From a String（python）




### 描述

You are given a string s, which contains stars *. In one operation, you can:

* Choose a star in s.
* Remove the closest non-star character to its left, as well as remove the star itself.

Return the string after all stars have been removed.

Note:

* The input will be generated such that the operation is always possible.
* It can be shown that the resulting string will always be unique.



Example 1:

	Input: s = "leet**cod*e"
	Output: "lecoe"
	Explanation: Performing the removals from left to right:
	- The closest character to the 1st star is 't' in "leet**cod*e". s becomes "lee*cod*e".
	- The closest character to the 2nd star is 'e' in "lee*cod*e". s becomes "lecod*e".
	- The closest character to the 3rd star is 'd' in "lecod*e". s becomes "lecoe".
	There are no more stars, so we return "lecoe".

	
Example 2:

	Input: s = "erase*****"
	Output: ""
	Explanation: The entire string is removed, so we return an empty string.




Note:

	1 <= s.length <= 10^5
	s consists of lowercase English letters and stars *.
	The operation above can be performed on s.


### 解析

根据题意，给定一个字符串 s ，其中包含星号 * 。 在一次操作中，我们可以：

* 在 s 中选择一颗星号
* 移除最靠近其左侧的非星形字符，以及移除星形本身

移除所有星号后返回字符串。

注意：

* 生成的输入保证始终可以进行上述操作
* 可以证明生成的字符串总是唯一的


其实这道题很简单，主要考察的是数据结构——栈的使用，我们直接遍历字符串 s ，当字符不为星号的时候，我们将该字符拼接到结果 result 后面，如果当前的字符为星号的时候，如果现有的 result 不为空，我们就将最后一个字符弹出去掉，这样遍历结束之后即可得到最后的结果 result ，直接返回 result 即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def removeStars(self, s):
	        """
	        :type s: str
	        :rtype: str
	        """
	        result = []
	        s = list(s)
	        for c in s:
	            if c == '*':
	                if result:
	                    result.pop()
	            else:
	                result.append(c)
	        return ''.join(result)
	        

### 运行结果

	
	65 / 65 test cases passed.
	Status: Accepted
	Runtime: 637 ms
	Memory Usage: 17.1 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-308/problems/removing-stars-from-a-string/


您的支持是我最大的动力
