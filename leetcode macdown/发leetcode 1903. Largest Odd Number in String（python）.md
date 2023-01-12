leetcode  1903. Largest Odd Number in String（python）

### 描述

You are given a string num, representing a large integer. Return the largest-valued odd integer (as a string) that is a non-empty substring of num, or an empty string "" if no odd integer exists.

A substring is a contiguous sequence of characters within a string.





Example 1:

	
	Input: num = "52"
	Output: "5"
	Explanation: The only non-empty substrings are "5", "2", and "52". "5" is the only odd number.
	
Example 2:


	Input: num = "4206"
	Output: ""
	Explanation: There are no odd numbers in "4206".

Example 3:

	Input: num = "35427"
	Output: "35427"
	Explanation: "35427" is already an odd number.

	



Note:


	1 <= num.length <= 10^5
	num only consists of digits and does not contain any leading zeros.

### 解析


根据题意，就是找出可以转换成最大奇数的 num 子字符串，如果没有就返回了空子符串。思路很简单，因为判断奇数，只需要判断最后一个字符串表示的数字是否是奇数即可，所以只需要从后向前遍历 num 中的字符，如果索引 i 上的数字为奇数，则直接返回 num[:i+1] ，遍历结束如果没有找到直接返回空字符串。

### 解答
				

	class Solution(object):
	    def largestOddNumber(self, num):
	        """
	        :type num: str
	        :rtype: str
	        """
	        result = ""
	        N = len(num)
	        for i in range(N-1,-1,-1):
	            if int(num[i])%2 :
	                return num[:i+1]
	        return result
	

            	      
			
### 运行结果

	Runtime: 72 ms, faster than 52.06% of Python online submissions for Largest Odd Number in String.
	Memory Usage: 18.7 MB, less than 29.38% of Python online submissions for Largest Odd Number in String.

### 解析

根据题意，其实上面判断的就是最后一个数字是不是奇数，可以换一种思路，就是只要让字符串的最后一个数字成为奇数就可以了，代码实现起来就是直接使用 python 的内置函数 num.rstrip 将最右边的偶数字符都去掉即可。很明显这种去字符的操作比判断字符的操作速度更快一点，而且使用内存也省了很多。

### 解答

	class Solution(object):
	    def largestOddNumber(self, num):
	        """
	        :type num: str
	        :rtype: str
	        """
	        return num.rstrip('02468')

### 运行结果

	Runtime: 29 ms, faster than 98.45% of Python online submissions for Largest Odd Number in String.
	Memory Usage: 15.3 MB, less than 98.45% of Python online submissions for Largest Odd Number in String.
	
原题链接：https://leetcode.com/problems/largest-odd-number-in-string



您的支持是我最大的动力
