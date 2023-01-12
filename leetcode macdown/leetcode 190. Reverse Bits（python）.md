leetcode  190. Reverse Bits（python）

### 描述
Reverse bits of a given 32 bits unsigned integer.

Note:

* Note that in some languages, such as Java, there is no unsigned integer type. In this case, both input and output will be given as a signed integer type. They should not affect your implementation, as the integer's internal binary representation is the same, whether it is signed or unsigned.
* In Java, the compiler represents the signed integers using 2's complement notation. Therefore, in Example 2 above, the input represents the signed integer -3 and the output represents the signed integer -1073741825.

Follow up: If this function is called many times, how would you optimize it?

Example 1:

	Input: n = 00000010100101000001111010011100
	Output:    964176192 (00111001011110000010100101000000)
	Explanation: The input binary string 00000010100101000001111010011100 represents the unsigned integer 43261596, so return 964176192 which its binary representation is 00111001011110000010100101000000.

	
Example 2:

	Input: n = 11111111111111111111111111111101
	Output:   3221225471 (10111111111111111111111111111111)
	Explanation: The input binary string 11111111111111111111111111111101 represents the unsigned integer 4294967293, so return 3221225471 which its binary representation is 10111111111111111111111111111111.



Note:

	The input must be a binary string of length 32


### 解析

根据题意，就是给定的 32 位无符号数的位。比如，给定输入整数 43261596（二进制表示为 00000010100101000001111010011100 ），返回 964176192（二进制表示为 00111001011110000010100101000000 ）。

题目还给我提出了更高的要求：如果该函数被多次调用，该如何优化？

最简单的就是使用内置函数，使用 bin 函数将 n 的二进制得到进行反转，然后


### 解答
				
	class Solution:
	    def reverseBits(self, n):
	        b = bin(n)[:1:-1]
	        return int(b + '0'*(32-len(b)), 2)

            	      
			
### 运行结果

	Runtime: 12 ms, faster than 98.13% of Python online submissions for Reverse Bits.
	Memory Usage: 13.2 MB, less than 97.31% of Python online submissions for Reverse Bits.
### 解析

根据题意，从后向前找出 n 的二进制的每个数字，然后可以使用位运算进行结果的计算：

* 循环 32 次
* 每次循环将结果变量 result 左移一位，相当于乘 2 ，如果当前的二进制数字为 1 ，那可以将 result 加 1 ，然后将 n 右移一位，相当于除 2
* 遍历结束得到 result 

### 解答

	class Solution:
	    def reverseBits(self, n):
	        result = 0
	        for _ in range(32):
	            result <<= 1
	            if n&1 :
	                result += 1
	            n >>= 1
	        return result

### 运行结果

	Runtime: 16 ms, faster than 91.61% of Python online submissions for Reverse Bits.
	Memory Usage: 13.6 MB, less than 10.57% of Python online submissions for Reverse Bits.
	
原题链接：https://leetcode.com/problems/reverse-bits/



您的支持是我最大的动力
