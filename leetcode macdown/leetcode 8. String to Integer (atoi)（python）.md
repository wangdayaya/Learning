leetcode  8. String to Integer (atoi)（python）

### 每日经典

《哥舒歌》 ——唐朝西部边民（唐）

北斗七星高，哥舒夜带刀。
 
至今窥牧马，不敢过临洮。

### 描述

Implement the myAtoi(string s) function, which converts a string to a 32-bit signed integer (similar to C/C++'s atoi function).

The algorithm for myAtoi(string s) is as follows:

* Read in and ignore any leading whitespace.
* Check if the next character (if not already at the end of the string) is '-' or '+'. Read this character in if it is either. This determines if the final result is negative or positive respectively. Assume the result is positive if neither is present.
* Read in next the characters until the next non-digit character or the end of the input is reached. The rest of the string is ignored.
* Convert these digits into an integer (i.e. "123" -> 123, "0032" -> 32). If no digits were read, then the integer is 0. Change the sign as necessary (from step 2).
* If the integer is out of the 32-bit signed integer range [-2<sup>31</sup>, 2<sup>31</sup> - 1], then clamp the integer so that it remains in the range. Specifically, integers less than -2<sup>31</sup> should be clamped to -2<sup>31</sup>, and integers greater than 2<sup>31</sup> - 1 should be clamped to 2<sup>31</sup> - 1.
* Return the integer as the final result.

Note:

* Only the space character ' ' is considered a whitespace character.
* Do not ignore any characters other than the leading whitespace or the rest of the string after the digits.




Example 1:


	Input: s = "42"
	Output: 42
	Explanation: The underlined characters are what is read in, the caret is the current reader position.
	Step 1: "42" (no characters read because there is no leading whitespace)
	         ^
	Step 2: "42" (no characters read because there is neither a '-' nor '+')
	         ^
	Step 3: "42" ("42" is read in)
	           ^
	The parsed integer is 42.
	Since 42 is in the range [-231, 231 - 1], the final result is 42.
	
Example 2:

	Input: s = "   -42"
	Output: -42
	Explanation:
	Step 1: "   -42" (leading whitespace is read and ignored)
	            ^
	Step 2: "   -42" ('-' is read, so the result should be negative)
	             ^
	Step 3: "   -42" ("42" is read in)
	               ^
	The parsed integer is -42.
	Since -42 is in the range [-231, 231 - 1], the final result is -42.


Example 3:

	Input: s = "4193 with words"
	Output: 4193
	Explanation:
	Step 1: "4193 with words" (no characters read because there is no leading whitespace)
	         ^
	Step 2: "4193 with words" (no characters read because there is neither a '-' nor '+')
	         ^
	Step 3: "4193 with words" ("4193" is read in; reading stops because the next character is a non-digit)
	             ^
	The parsed integer is 4193.
	Since 4193 is in the range [-231, 231 - 1], the final result is 4193.




Note:

* 	0 <= s.length <= 200
* 	s consists of English letters (lower-case and upper-case), digits (0-9), ' ', '+', '-', and '.'.



### 解析

根据题意，实现 myAtoi(string s) 函数，将字符串转换为 32 位有符号整数。myAtoi(string s) 的算法如下：

* 忽略任何前导空格
* 然后检查下一个字符（如果不在字符串末尾）是“-”还是“+”。如果是，请读入此字符确定最终结果是负数还是正数。如果两者都不存在，则假设结果是正数
* 读入下一个字符，直到到达下一个非数字字符或输入的结尾，字符串的其余部分被忽略。
* 将这些数字转换为整数（即​​“123”-> 123、“0032”-> 32）。如果未读取任何数字，则整数为 0。根据需要更改符号（从步骤 2 开始）。
* 如果整数超出 32 位有符号整数范围 [-2<sup>31</sup>, 2<sup>31</sup> - 1]，小于 -2<sup>31</sup> 的整数应限制为 -2<sup>31</sup>，大于 2<sup>31</sup> - 1 的整数应限制为 2<sup>31</sup> - 1。
* 返回整数作为最终结果。

其实这个题有很多边界的 case ，这是需要注意的，其他只需要按照上面的算法从前到后写出来即可，我觉的这道题 Acceptance 只有 16% ，可能就是很多人按照自己的思路写代码导致有很多边界 case 报错了，因为我就是这样【狗头】。

### 解答
				

	class Solution(object):
	    def myAtoi(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        if not s : return 0
	        sign = 1
	        s = s.lstrip()
	        if not s: return 0
	        if s[0] == "+" or s[0] == "-":
	            sign = 1 if s[0] == "+" else  -1 
	            s = s[1:]
	        print(s)
	        if not s : return 0
	        result = 0
	        for i in range(len(s)):
	            digit = ord(s[i])-ord('0')
	            if digit<0 or digit>9: break
	            result = result * 10 + digit
	    
	        result *= sign
	        if result>0:
	            return result if result<=pow(2,31)-1 else pow(2,31)-1
	        return result if result>-pow(2,31) else -pow(2,31)
	            
	        
	        
            	      
			
### 运行结果

	Runtime: 22 ms, faster than 69.11% of Python online submissions for String to Integer (atoi).
	Memory Usage: 13.6 MB, less than 52.93% of Python online submissions for String to Integer (atoi).


原题链接：https://leetcode.com/problems/string-to-integer-atoi/



您的支持是我最大的动力
