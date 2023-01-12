leetcode  1323. Maximum 69 Number（python）

### 描述



Given a positive integer num consisting only of digits 6 and 9.

Return the maximum number you can get by changing at most one digit (6 becomes 9, and 9 becomes 6).

Example 1:


	Input: num = 9669
	Output: 9969
	Explanation: 
	Changing the first digit results in 6669.
	Changing the second digit results in 9969.
	Changing the third digit results in 9699.
	Changing the fourth digit results in 9666. 
	The maximum number is 9969.
	
Example 2:


	Input: num = 9996
	Output: 9999
	Explanation: Changing the last digit 6 to 9 results in the maximum number.

Example 3:


	Input: num = 9999
	Output: 9999
	Explanation: It is better not to apply any change.
	


Note:

	1 <= num <= 10^4
	num's digits are 6 or 9.


### 解析


根据题意，就是经过一次将 6 变 9 或者 9 变 6 的操作，能得到的最大数字是多少。只需要遍历每个字符，每次执行一次上述的“69”操作，然后和 num 进行大小比较，得到较大数赋值给 num ，遍历结束即可得到最大的数字。

### 解答
				
	class Solution(object):
	    def maximum69Number (self, num):
	        """
	        :type num: int
	        :rtype: int
	        """
	        s = str(num)
	        for i in range(len(s)):
	            if s[i]=='9':
	                num = max(int(s[:i]+'6'+s[i+1:]), num)
	            else:
	                num = max(int(s[:i]+'9'+s[i+1:]), num)
	        return num
	


            	      
			
### 运行结果

	Runtime: 24 ms, faster than 5.10% of Python online submissions for Maximum 69 Number.
	Memory Usage: 13.6 MB, less than 9.44% of Python online submissions for Maximum 69 Number.

### 解析


如果吃透题目，其实最大的数，无非就是将从前到后的第一个出现的 6 变为 9 就能得到最大数。

### 解答


	class Solution(object):
	    def maximum69Number (self, num):
	        """
	        :type num: int
	        :rtype: int
	        """
	        return str(num).replace('6','9',1)


### 运行结果

	
	Runtime: 16 ms, faster than 69.39% of Python online submissions for Maximum 69 Number.
	Memory Usage: 13.1 MB, less than 100.00% of Python online submissions for Maximum 69 Number.


原题链接：https://leetcode.com/problems/maximum-69-number/



您的支持是我最大的动力
