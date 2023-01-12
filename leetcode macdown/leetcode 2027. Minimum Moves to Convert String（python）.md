leetcode  2027. Minimum Moves to Convert String（python）

### 描述

You are given a string s consisting of n characters which are either 'X' or 'O'.

A move is defined as selecting three consecutive characters of s and converting them to 'O'. Note that if a move is applied to the character 'O', it will stay the same.

Return the minimum number of moves required so that all the characters of s are converted to 'O'.



Example 1:


	Input: s = "XXX"
	Output: 1
	Explanation: XXX -> OOO
	We select all the 3 characters and convert them in one move.
	
Example 2:

	Input: s = "XXOX"
	Output: 2
	Explanation: XXOX -> OOOX -> OOOO
	We select the first 3 characters in the first move, and convert them to 'O'.
	Then we select the last 3 characters and convert them so that the final string contains all 'O's.


Example 3:


	Input: s = "OOOO"
	Output: 0
	Explanation: There are no 'X's in s to convert.
	



Note:


	3 <= s.length <= 1000
	s[i] is either 'X' or 'O'.

### 解析

根据题意，就是给出了只包含 X 或者 O 的字符串 s ，要求我们找出最少的操作次数，可以将 s 中的字符全部都变为 O 字符，每次操作的过程如下：每次选中三个连续的字符，如果字符为 X  则转为 O ，如果字符为 O 则不变。

其实这道题看起来有点摸不着头脑，其实想明白了还是挺简单的：

* 初始化一个结果 result 为 0 ，索引 i 为 0
* 当 i<len(s) 的时候，如果 s[i] 为 O ， 则不用进行转换操作，直接 i 加一对下一个字符进行判断，否则当  s[i] 为 X 的时候，不管 X 之后的两个字符是什么，肯定要进行一次操作，将三个字符都变成 O ，然后 i 加三对下一个字符进行判断
* 循环结束之后返回 result 



### 解答
				

	class Solution(object):
	    def minimumMoves(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        result = 0
	        i = 0
	        while i<len(s):
	            if s[i]=='O':
	                i += 1
	                continue
	            else:
	                result += 1
	                i += 3
	        return result
	      
			
### 运行结果

	Runtime: 16 ms, faster than 93.99% of Python online submissions for Minimum Moves to Convert String.
	Memory Usage: 13.3 MB, less than 99.05% of Python online submissions for Minimum Moves to Convert String.



原题链接：https://leetcode.com/problems/minimum-moves-to-convert-string/



您的支持是我最大的动力
