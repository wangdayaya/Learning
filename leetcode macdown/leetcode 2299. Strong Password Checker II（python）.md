leetcode 2299. Strong Password Checker II （python）




### 描述

A password is said to be strong if it satisfies all the following criteria:

* It has at least 8 characters.
* It contains at least one lowercase letter.
* It contains at least one uppercase letter.
* It contains at least one digit.
* It contains at least one special character. The special characters are the characters in the following string: "!@#$%^&*()-+".
* It does not contain 2 of the same character in adjacent positions (i.e., "aab" violates this condition, but "aba" does not).
* Given a string password, return true if it is a strong password. Otherwise, return false.



Example 1:

	Input: password = "IloveLe3tcode!"
	Output: true
	Explanation: The password meets all the requirements. Therefore, we return true.

	
Example 2:

	Input: password = "Me+You--IsMyDream"
	Output: false
	Explanation: The password does not contain a digit and also contains 2 of the same character in adjacent positions. Therefore, we return false.


Example 3:

	Input: password = "1aB!"
	Output: false
	Explanation: The password does not meet the length requirement. Therefore, we return false.

	





Note:


	1 <= password.length <= 100
	password consists of letters, digits, and special characters: "!@#$%^&*()-+".

### 解析

根据题意，如果满足以下所有条件，则称密码为强密码：

* 它至少有 8 个字符。
* 它至少包含一个小写字母。
* 它至少包含一个大写字母。
* 它至少包含一位数字。
* 它至少包含一个特殊字符。 特殊字符是以下字符串中的字符：“!@#$%^&*()-+”。
* 它在相邻位置不包含 2 个相同的字符（即，“aab”违反了此条件，但“aba”没有）。

给定一个字符串密码，如果它是一个强密码，则返回 true。 否则，返回 False 。

这道题其实就是考查对字符串的操作，最简单的就是使用正则进行解题，我直接用了最暴力的方法，不用费脑子。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。


### 解答
				

	class Solution(object):
	    def strongPasswordCheckerII(self, password):
	        """
	        :type password: str
	        :rtype: bool
	        """
	        counter = collections.Counter(list(password))
	        keys = set(counter.keys())
	        if len(password) < 8:
	            return False
	        if not keys & set(list('qwertyuiopasdfghjklzxcvbnm')):
	            return False
	        if not keys & set(list('QWERTYUIOPASDFGHJKLZXCVBNM')):
	            return False
	        if not keys & set(list('1234567890')):
	            return False
	        if not keys & set(list('!@#$%^&*()-+')):
	            return False
	        for i in range(1, len(password)):
	            if password[i-1] == password[i]:
	                return False
	        return True
            	      
			
### 运行结果



	148 / 148 test cases passed.
	Status: Accepted
	Runtime: 17 ms
	Memory Usage: 13.5 MB

### 原题链接

https://leetcode.com/contest/biweekly-contest-80/problems/strong-password-checker-ii/

您的支持是我最大的动力
