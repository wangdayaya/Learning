

### 描述
Every non-negative integer N has a binary representation.  For example, 5 can be represented as "101" in binary, 11 as "1011" in binary, and so on.  Note that except for N = 0, there are no leading zeroes in any binary representation.

The complement of a binary representation is the number in binary you get when changing every 1 to a 0 and 0 to a 1.  For example, the complement of "101" in binary is "010" in binary.

For a given number N in base-10, return the complement of it's binary representation as a base-10 integer.

 

Example 1:

	Input: 5
	Output: 2
	Explanation: 5 is "101" in binary, with complement "010" in binary, which is 2 in base-10.
Example 2:

	Input: 7
	Output: 0
	Explanation: 7 is "111" in binary, with complement "000" in binary, which is 0 in base-10.
Example 3:

	Input: 10
	Output: 5
	Explanation: 10 is "1010" in binary, with complement "0101" in binary, which is 5 in base-10.
 

Note:

	0 <= N < 10^9
### 解析
根据题意，就是把 N 变成二进制的形态，然后将 1 变成 0 ，0 变成 1 ，将变换后的数字再转换回十进制。时间复杂度为 O(N)，空间复杂度为 O(N)，N 是二进制的位数。可以直接使用 python 的内置函数，按照上面的思路进行即可。

### 解答
				
	class Solution(object):
    def bitwiseComplement(self, N):
        """
        :type N: int
        :rtype: int
        """
        result = ""
        nbins = bin(N)[2:]
        for i in nbins:
            result += str(int(i)^1)
        return int(result, 2)            
                     	      
			
### 运行结果
	Runtime: 16 ms, faster than 77.81% of Python online submissions for Complement of Base 10 Integer.
	Memory Usage: 11.7 MB, less than 79.49% of Python online submissions for Complement of Base 10 Integer.


### 解析

根据题意，我们还可以进行二进制的位运算和加法运算，先拿到 n 的二进制字符串 b ，然后从后向前一位一位计算补码数字的十进制大小，将其加入结果 result ，最后返回即可。

### 解答


	class Solution(object):
	    def bitwiseComplement(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        b = bin(n)[2:]
	        result = 0
	        for i in range(len(b)-1, -1, -1):
	            result += (int(b[i])^1) * pow(2, len(b)-1-i)
	        return result
	        
### 运行结果
	
	Runtime: 20 ms, faster than 40.00% of Python online submissions for Complement of Base 10 Integer.
	Memory Usage: 13.4 MB, less than 56.84% of Python online submissions for Complement of Base 10 Integer.
	
每日格言：生命不等于是呼吸，生命是活动。

