leetcode  1945. Sum of Digits of String After Convert（python）

### 描述

You are given a string s consisting of lowercase English letters, and an integer k.

First, convert s into an integer by replacing each letter with its position in the alphabet (i.e., replace 'a' with 1, 'b' with 2, ..., 'z' with 26). Then, transform the integer by replacing it with the sum of its digits. Repeat the transform operation k times in total.

For example, if s = "zbax" and k = 2, then the resulting integer would be 8 by the following operations:

* Convert: "zbax" ➝ "(26)(2)(1)(24)" ➝ "262124" ➝ 262124
* Transform #1: 262124 ➝ 2 + 6 + 2 + 1 + 2 + 4 ➝ 17
* Transform #2: 17 ➝ 1 + 7 ➝ 8

Return the resulting integer after performing the operations described above.

 



Example 1:

	Input: s = "iiii", k = 1
	Output: 36
	Explanation: The operations are as follows:
	- Convert: "iiii" ➝ "(9)(9)(9)(9)" ➝ "9999" ➝ 9999
	- Transform #1: 9999 ➝ 9 + 9 + 9 + 9 ➝ 36
	Thus the resulting integer is 36.

	
Example 2:

	Input: s = "leetcode", k = 2
	Output: 6
	Explanation: The operations are as follows:
	- Convert: "leetcode" ➝ "(12)(5)(5)(20)(3)(15)(4)(5)" ➝ "12552031545" ➝ 12552031545
	- Transform #1: 12552031545 ➝ 1 + 2 + 5 + 5 + 2 + 0 + 3 + 1 + 5 + 4 + 5 ➝ 33
	- Transform #2: 33 ➝ 3 + 3 ➝ 6
	Thus the resulting integer is 6.


Example 3:

	Input: s = "zbax", k = 2
	Output: 8

	

Note:


	1 <= s.length <= 100
	1 <= k <= 10
	s consists of lowercase English letters.

### 解析

根据题意，就是给出了一个都是小写英文字母的字符串 s ，和一个整数 k 。

首先我们要先把字符串 s 中的每个字母都转换成数字，例如 a 转为 1 ，b 转为 2 ，以此类推。然后将得到的数字从左到右拼接成字符串，再将字符串中的每个数字相加得到和替换该字符串，执行这样的操作 k 次，返回最后得到的结果。思路很简单：

* 初始化结果 result 为空字符串
* 将 s 中的每个字符转换成数字字符串按顺序拼接在 result 之后
* while 循环 k 次，将每个字符变成数字相加之后得到的和，再变为字符串赋给 result ，同时 k 减一 ，继续循环该操作
* 循环结果返回 result 


### 解答
				
	class Solution(object):
	    def getLucky(self, s, k):
	        """
	        :type s: str
	        :type k: int
	        :rtype: int
	        """
	        result = ""
	        for i in s:
	            result += str(ord(i)-96)
	        while k:
	            result = str(sum([int(c) for c in result]))
	            k-=1
	        return result
	        
	        

            	      
			
### 运行结果


	Runtime: 20 ms, faster than 96.63% of Python online submissions for Sum of Digits of String After Convert.
	Memory Usage: 13.6 MB, less than 10.11% of Python online submissions for Sum of Digits of String After Convert.

原题链接：https://leetcode.com/problems/sum-of-digits-of-string-after-convert/



您的支持是我最大的动力
