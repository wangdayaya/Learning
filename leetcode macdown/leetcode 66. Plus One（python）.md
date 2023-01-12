leetcode 66. Plus One （python）

### 描述



You are given a large integer represented as an integer array digits, where each digits[i] is the ith digit of the integer. The digits are ordered from most significant to least significant in left-to-right order. The large integer does not contain any leading 0's.

Increment the large integer by one and return the resulting array of digits.

Example 1:


	Input: digits = [1,2,3]
	Output: [1,2,4]
	Explanation: The array represents the integer 123.
	Incrementing by one gives 123 + 1 = 124.
	Thus, the result should be [1,2,4].
	
Example 2:


	Input: digits = [4,3,2,1]
	Output: [4,3,2,2]
	Explanation: The array represents the integer 4321.
	Incrementing by one gives 4321 + 1 = 4322.
	Thus, the result should be [4,3,2,2].

Example 3:


	Input: digits = [0]
	Output: [1]
	Explanation: The array represents the integer 0.
	Incrementing by one gives 0 + 1 = 1.
	Thus, the result should be [1].
	
Example 4:

	Input: digits = [9]
	Output: [1,0]
	Explanation: The array represents the integer 9.
	Incrementing by one gives 9 + 1 = 10.
	Thus, the result should be [1,0].



Note:

	1 <= digits.length <= 100
	0 <= digits[i] <= 9
	digits does not contain any leading 0's.


### 解析

根据题意，给出了一个列表表示一个整数，然后给这个“整数”加一，然后返回所得结果的列表形式。其实思路比较简单，就是模拟正常的整数相加规律，从后往前相加，如果满 10 就往前一位进一，需要注意的是如果在最左边的元素为 10 ，需要特殊处理。


### 解答
				
	class Solution(object):
	    def plusOne(self, digits):
	        """
	        :type digits: List[int]
	        :rtype: List[int]
	        """
	        digits[-1] += 1
	        for i in range(len(digits)-1, 0, -1):
	            if digits[i] != 10:
	                break
	            digits[i] = 0
	            digits[i-1] += 1
	
	        if digits[0] == 10:
	            digits[0] = 0
	            return [1] + digits
	        return digits
            	      
			
### 运行结果

	Runtime: 39 ms, faster than 6.82% of Python online submissions for Plus One.
	Memory Usage: 13.2 MB, less than 92.25% of Python online submissions for Plus One.


### 解析

当然还可以使用内置函数，先将 digits 换算成整数然后加一，最后将结果用列表的形式展示出来即可。

### 解答

	class Solution(object):
	    def plusOne(self, digits):
	        """
	        :type digits: List[int]
	        :rtype: List[int]
	        """
	        num = 0
	        N = len(digits)
	        for i, digit in enumerate(digits):
	            num += pow(10, N-i-1) * digit
	        return [int(i) for i in str(num+1)]

### 运行结果

	Runtime: 36 ms, faster than 10.12% of Python online submissions for Plus One.
	Memory Usage: 13.5 MB, less than 42.47% of Python online submissions for Plus One.

### 解析

看了官网的大神解答，才发现还能用递归进行求解，是我格局小了。

### 解答
	
	class Solution(object):
	    def plusOne(self, digits):
	        """
	        :type digits: List[int]
	        :rtype: List[int]
	        """
	        if digits[-1] + 1 < 10:
	            digits[-1] += 1
	            return digits
	        elif len(digits) == 1 and digits[-1] + 1 == 10:
	            return [1, 0]
	        else:
	            digits[-1] = 0
	            digits[:-1] = self.plusOne(digits[:-1])
	            return digits

### 运行结果

	Runtime: 32 ms, faster than 15.85% of Python online submissions for Plus One.
	Memory Usage: 13.4 MB, less than 42.47% of Python online submissions for Plus One.

原题链接：https://leetcode.com/problems/plus-one/



您的支持是我最大的动力
