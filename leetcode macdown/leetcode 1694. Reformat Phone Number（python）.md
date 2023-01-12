leetcode  1694. Reformat Phone Number（python）

### 描述

You are given a phone number as a string number. number consists of digits, spaces ' ', and/or dashes '-'.

You would like to reformat the phone number in a certain manner. Firstly, remove all spaces and dashes. Then, group the digits from left to right into blocks of length 3 until there are 4 or fewer digits. The final digits are then grouped as follows:

* 2 digits: A single block of length 2.
* 3 digits: A single block of length 3.
* 4 digits: Two blocks of length 2 each.

The blocks are then joined by dashes. Notice that the reformatting process should never produce any blocks of length 1 and produce at most two blocks of length 2.

Return the phone number after formatting.





Example 1:

	Input: number = "1-23-45 6"
	Output: "123-456"
	Explanation: The digits are "123456".
	Step 1: There are more than 4 digits, so group the next 3 digits. The 1st block is "123".
	Step 2: There are 3 digits remaining, so put them in a single block of length 3. The 2nd block is "456".
	Joining the blocks gives "123-456".

	
Example 2:

	Input: number = "123 4-567"
	Output: "123-45-67"
	Explanation: The digits are "1234567".
	Step 1: There are more than 4 digits, so group the next 3 digits. The 1st block is "123".
	Step 2: There are 4 digits left, so split them into two blocks of length 2. The blocks are "45" and "67".
	Joining the blocks gives "123-45-67".


Example 3:

	Input: number = "123 4-5678"
	Output: "123-456-78"
	Explanation: The digits are "12345678".
	Step 1: The 1st block is "123".
	Step 2: The 2nd block is "456".
	Step 3: There are 2 digits left, so put them in a single block of length 2. The 3rd block is "78".
	Joining the blocks gives "123-456-78".

	
Example 4:


	Input: number = "12"
	Output: "12"
	
Example 5:


	Input: number = "--17-5 229 35-39475 "
	Output: "175-229-353-94-75"

Note:


	2 <= number.length <= 100
	number consists of digits and the characters '-' and ' '.
	There are at least two digits in number.

### 解析

根据题意，就是将先将 number 中的空格和 - 去掉，然后每 3 位分成一小块，用 - 连接起来，如果最后剩下三位或者两位数字，则直接用 - 连接在之前的字符串后面，如果剩下的是四位，则用两个两位数字和 - 连接在之前的字符串后面，最后返回用 - 连接好的字符串。


### 解答
				
	
	class Solution(object):
	    def reformatNumber(self, number):
	        """
	        :type number: str
	        :rtype: str
	        """
	        number = number.replace(' ','').replace('-','')
	        def f(number, r):
	            if len(number)<=4:
	                if len(number)==2 or len(number)==3:
	                    return r+number
	                else:
	                    return r+number[:2]+'-'+number[2:]
	            r += number[:3] + '-'
	            return f(number[3:], r)
	        return f(number, '')
            	      
			
### 运行结果

	Runtime: 16 ms, faster than 75.00% of Python online submissions for Reformat Phone Number.
	Memory Usage: 13.5 MB, less than 37.50% of Python online submissions for Reformat Phone Number.




原题链接：https://leetcode.com/problems/reformat-phone-number/submissions/



您的支持是我最大的动力
