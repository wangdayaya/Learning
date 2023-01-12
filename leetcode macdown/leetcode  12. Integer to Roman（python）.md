leetcode 12. Integer to Roman （python）




### 描述

Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.

	Symbol       Value
	I             1
	V             5
	X             10
	L             50
	C             100
	D             500
	M             1000

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

* I can be placed before V (5) and X (10) to make 4 and 9. 
* X can be placed before L (50) and C (100) to make 40 and 90. 
* C can be placed before D (500) and M (1000) to make 400 and 900.

Given an integer, convert it to a roman numeral.

Example 1:


	Input: num = 3
	Output: "III"
	Explanation: 3 is represented as 3 ones.
	
Example 2:


	Input: num = 58
	Output: "LVIII"
	Explanation: L = 50, V = 5, III = 3.

Example 3:



	Input: num = 1994
	Output: "MCMXCIV"
	Explanation: M = 1000, CM = 900, XC = 90 and IV = 4.

Note:

	1 <= num <= 3999


### 解析

根据题意，罗马数字由七个不同的符号表示：I，V，X，L，C，D 和 M 。代表的值如下所示：

	I             1
	V             5
	X             10
	L             50
	C             100
	D             500
	M             1000
	
罗马数字通常从左到右书写从大到小。但是 4 的罗马数字不是 IIII 而是 IV 。因为在 5 之前我们减去 1 就变成 4 。同样的原则也适用于数字 9 ，它被写成 IX 。有六种情况适用：

* I 可以放在 V（5）和 X（10）之前，得出 4 和 9 。
* X 可以放在 L（50）和 C（100）之前，得出 40 和 90 。
* C 可以放在 D（500）和 M（1000）之前，制成 400 和 900 。

给定一个整数，将其转变成罗马数字字符串。

这道题考查的就是贪心思想，因为对于一个数字我们都尽量从做往右尽量取较大数字的罗马字符串进行表示，所以我们可以先定义一个保存了所有阿拉伯数字到罗马数字的字典 d ，然后将可用的阿拉伯数字都降序放到列表 L 中：

while 循环处理 num 即可，当 num 大于 0 的时候，遍历 L 中的每个数字 k ，如果 num 整除 k 大于 0 ，说明 num 对应的罗马表示的最左边可以用 k 对应的罗马数字进行表示，将其拼接到结果 result 后面，然后将 num 减去 k ，跳出遍历，继续执行下一次 while 循环，直到 num 为 0 跳出循环，最后得到的 result 就是答案。

时间复杂度为 O(N\*M) ，N 是 num 阿拉伯数字个数，题目中规定 num 最大为 3999 ，N 最大为 4 ， M 是 L 的长度，M 最大为 13 ，所以时间复杂度为也可以写成 O(1) ，空间复杂度为 O(M) ，同理也可以写成 O(1) 。

### 解答

	class Solution(object):
	    def intToRoman(self, num):
	        """
	        :type num: int
	        :rtype: str
	        """
	        d = {1: 'I', 5: 'V', 10: 'X', 50: 'L', 100: 'C', 500: 'D', 1000: 'M',
	         4: 'IV', 9: 'IX', 40: 'XL', 90: 'XC', 400: 'CD', 900: 'CM'}
	        L = sorted(d.keys(),reverse=True)
	        result = ''
	        while num > 0:
	            for k in L:
	                if num // k > 0:
	                    result += d[k]
	                    num -= k
	                    break
	        return result

### 运行结果

	Runtime: 98 ms, faster than 20.22% of Python online submissions for Integer to Roman.
	Memory Usage: 13.2 MB, less than 87.76% of Python online submissions for Integer to Roman.

### 原题链接

https://leetcode.com/problems/integer-to-roman/


您的支持是我最大的动力
