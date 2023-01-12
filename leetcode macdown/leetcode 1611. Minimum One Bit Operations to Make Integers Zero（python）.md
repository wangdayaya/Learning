leetcode  1611. Minimum One Bit Operations to Make Integers Zero（python）

### 描述


Given an integer n, you must transform it into 0 using the following operations any number of times:

* Change the rightmost (0<sup>th</sup>) bit in the binary representation of n.
* Change the i<sup>th</sup> bit in the binary representation of n if the (i-1)<sup>th</sup> bit is set to 1 and the (i-2)<sup>th</sup> through 0<sup>th</sup> bits are set to 0.

Return the minimum number of operations to transform n into 0.




Example 1:

	Input: n = 0
	Output: 0

	
Example 2:

	Input: n = 3
	Output: 2
	Explanation: The binary representation of 3 is "11".
	"11" -> "01" with the 2nd operation since the 0th bit is 1.
	"01" -> "00" with the 1st operation.


Example 3:

	Input: n = 6
	Output: 4
	Explanation: The binary representation of 6 is "110".
	"110" -> "010" with the 2nd operation since the 1st bit is 1 and 0th through 0th bits are 0.
	"010" -> "011" with the 1st operation.
	"011" -> "001" with the 2nd operation since the 0th bit is 1.
	"001" -> "000" with the 1st operation.

	
Example 4:

	Input: n = 9
	Output: 14

	
Example 5:


	Input: n = 333
	Output: 393

Note:


	0 <= n <= 109

### 解析

根据题意，给定一个整数 n，必须使用任意次数的以下操作将其转换为 0：

* 更改 n 的二进制表示最右边的位。可以 0 变为 1 ，也可以 1 变为 0 。
* 如果第 (i-1) 位设置为 1 并且第 (i-2) 到第 0 位设置为 0，则更改 n 的二进制表示中的第 i 位。可以 0 变为 1 ，也可以 1 变为 0 。

注意题目中的二进制位的索引都是从右向左的，返回将 n 转换为 0 的最小操作数。因为方法一只是将最右边的 0 和 1 互换，无法对前面的字符进行操作，所以关键就是巧用方法二进行变化，假如我们举例，将 101011 变为 000000 ，其最简单的思路就是递归：

* （1）101011 第一位为 1 ，想要将其变为 100000 ，就调用自定义的 convert 函数，该函数的功能就是找出将 01011 变为 10000 的最少次数
* （2）应用方法二将变化之后的 110000 变为 010000 进行了 1 次操作，然后计算将 10000 变为 00000 的次数，和上面同样的方法，将 0000 通过 convert 函数变为 1000 ，在进行相同的操作，直到最后变为 000000 
* （3）所以定义递归函数 dfs ，表示对输入二进制的最少次数操作，将上面的过程表示出来就是 dfs(101011) = convert(01011) + 1 + dfs(10000)

但是 convert 有两种情况：

* 第一种情况是二进制的第一个数字是 1 ，如 1110 。那直接调用 dfs(110) 即可
* 第二种情况是二进制的第一个数字是 0 ，如 0111 ，又是需要递归 ：convert(0111) = convert(111) + 1 + dfs(100)

### 解答
				

	class Solution(object):
	    
	    def __init__(self):
	        self.d = {}
	        self.t = {}
	        
	    def minimumOneBitOperations(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        return(self.dfs(bin(n)[2:]))
	    
	    def dfs(self, s):
	        if s == '0' : return 0
	        if s == '1' : return 1
	        if s in self.d : return self.d[s]
	        if  s[0] == '0': return self.dfs(s[1:])
	        m = s[1:]
	        n = list(s[1:])
	        n[0] = '1'
	        for i in range(1, len(n)):
	            n[i] = '0'
	        n = ''.join(n)
	        self.d[s] = self.convert(m) + 1 + self.dfs(n)
	        return self.d[s]
	    
	    def convert(self, s):
	        if s == '0' : return 1
	        if s == '1' : return 0
	        if s in self.t : return self.t[s]
	        if s[0] == '1': 
	            self.t[s] = self.dfs(s[1:])
	        else:
	            m = s[1:]
	            n = list(s[1:])
	            n[0] = '1'
	            for i in range(1, len(n)):
	                n[i] = '0'
	            n = ''.join(n)
	            self.t[s] = self.convert(m) + 1 + self.dfs(n)
	        return self.t[s]
	
	        
	                    	      
			
### 运行结果

	
	Runtime: 44 ms, faster than 5.17% of Python online submissions for Minimum One Bit Operations to Make Integers Zero.
	Memory Usage: 13.3 MB, less than 77.59% of Python online submissions for Minimum One Bit Operations to Make Integers Zero.

原题链接：https://leetcode.com/problems/minimum-one-bit-operations-to-make-integers-zero/



您的支持是我最大的动力
