leetcode  476. Number Complement（python）

### 描述


Given a positive integer num, output its complement number. The complement strategy is to flip the bits of its binary representation.


Example 1:

	Input: num = 5
	Output: 2
	Explanation: The binary representation of 5 is 101 (no leading zero bits), and its complement is 010. So you need to output 2.

	
Example 2:


	Input: num = 1
	Output: 0
	Explanation: The binary representation of 1 is 1 (no leading zero bits), and its complement is 0. So you need to output 0.




Note:

* The given integer num is guaranteed to fit within the range of a 32-bit signed integer.
* num >= 1
* You could assume no leading zero bit in the integer’s binary representation.
* This question is the same as 1009: https://leetcode.com/problems/complement-of-base-10-integer/


### 解析

根据题意，就是先找出 num 的二进制表达，然后求将其中的 1 变成 0 ，0 变成 1 形成的二进制数的十进制数。直接使用内置函数 bin 得到 num 的二进制字符串，然后遍历字符串进行上述的变化操作，最后将二进制转换成十进制返回即可。这种方法有使用内置函数的嫌疑，技术含量不高。


### 解答
				
	class Solution(object):
	    def findComplement(self, num):
	        """
	        :type num: int
	        :rtype: int
	        """
	        s = bin(num)[2:]
	        r = ''
	        for c in s:
	            if c=='0':
	                r += '1'
	            else:
	                r += '0'
	        return int(r,2)
	        

            	      
			
### 运行结果

	Runtime: 20 ms, faster than 38.33% of Python online submissions for Number Complement.
	Memory Usage: 13.2 MB, less than 84.58% of Python online submissions for Number Complement.

### 解析
因为这里需要将 0 变成 1 ，将 1 变成 0 ，最直接的我们想到了异或的操作，只要将二进制的每一位都和 1 近行异或，就能达到相同的效果。构建同样长度的二进制，用到了 << 位移操作，(1 << (len(bin(num)) - 2)) - 1 即可得到。然后将其和 num 进行 ^ 运算即可得到答案，这种方法比上面的方法要快很多。

异或运算：操作的是二进制位，两个位的数字相同为0，两个位的数字相异为1

	print(1^0) 
	print(1^1)
	print(0^0)
	
打印：

	1
	0
	0

<< 左移位运算：操作的是二进制位，各二进位全部左移若干位，高位丢弃，低位补0

	print(1)
	print(1<<2)
	print(1<<3)
	
打印：

	1
	4
	8


### 解答

	class Solution(object):
	    def findComplement(self, num):
	        """
	        :type num: int
	        :rtype: int
	        """
	        return num ^ ((1 << (len(bin(num)) - 2)) - 1)
	        

### 运行结果

	Runtime: 8 ms, faster than 98.74% of Python online submissions for Number Complement.
	Memory Usage: 13.3 MB, less than 88.05% of Python online submissions for Number Complement.

原题链接：https://leetcode.com/problems/number-complement/



您的支持是我最大的动力
