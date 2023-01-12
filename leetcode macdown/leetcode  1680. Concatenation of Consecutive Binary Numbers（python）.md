leetcode  1680. Concatenation of Consecutive Binary Numbers（python）




### 描述

Given an integer n, return the decimal value of the binary string formed by concatenating the binary representations of 1 to n in order, modulo 10^9 + 7.



Example 1:

	Input: n = 1
	Output: 1
	Explanation: "1" in binary corresponds to the decimal value 1.

	
Example 2:

	Input: n = 3
	Output: 27
	Explanation: In binary, 1, 2, and 3 corresponds to "1", "10", and "11".
	After concatenating them, we have "11011", which corresponds to the decimal value 27.


Example 3:

	Input: n = 12
	Output: 505379714
	Explanation: The concatenation results in "1101110010111011110001001101010111100".
	The decimal value of that is 118505380540.
	After modulo 109 + 7, the result is 505379714.



Note:


	1 <= n <= 10^5

### 解析

根据题意，给定一个整数 n ，返回由 1 到 n 的二进制表示按顺序从左到右连接形成的二进制字符串对应的的十进制值，结果可能很大，要取模 10^9 + 7 。

我们可以取一些数字挨个拼接一下，先找出规律，假如从 1 开始遍历：

* 最开始的字符串结果就是 1 
* 遍历到 2 的时候，需要把 1 的二进制 1 左移 2 的二进制位数，也就是两位，此时把 1 和 10 进行字符串拼接得到 110
* 遍历到 3 的时候，需要把 110  左移 3 的二进制位数，也就是两位，此时把 110 和 11 进行字符串拼接得到 11011 
* 遍历到 4 的时候，同样的道理，先左移之前的二进制若干位，然后拼接 4 的二进制 100 得到 11011100 
* 以此类推

我们发现假如 1 到 n-1 的二进制字符串连接结果是 x ，整数 n 的二进制有 Len<sub>2</sub>(n) 位，则将 x 左移 Len<sub>2</sub>(n) 位 之后把 x 拼接在后边，即可得到最后的二进制结果，也就能得到最后的十进制结果。

其实如果为了解题比赛，这里直接使用 python 的内置函数 bin 可以直接算出来每个整数的二进制位数，这样题目也能 AC ，但是这里我们还是使用位运算来计算每个整数的二进制位数。

假如 n 为 2 的整数次幂，则 n 比 n-1 的二进制表示的位数多 1 ，那么我们定义最开始的整数 0 的二进制长度 L 为 0 ，那么当 x & (x−1) 等于 0 的时候，说明 x 达到了 2 的整数次幂，L 应该加一，否则不尽兴加一操作，这样我们就能通过不断操作 L 来确定每个整数的二进制长度，这个操作的时间复杂度为 O(1)。

需要注意的是可能过程中值太多，结果需要不断取模，整个过程的时间复杂度为 O(N \* 1) ，空间复杂度为 O(1) 。



### 解答

	class Solution(object):
	    def concatenatedBinary(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        mod = 10**9 + 7
	        result = L = 0
	        for i in range(1, n + 1):
	            if (i & (i - 1)) == 0:
	                L += 1
	            result = ((result << L) + i) % mod
	        return result

### 运行结果

	Runtime: 828 ms, faster than 82.61% of Python online submissions for Concatenation of Consecutive Binary Numbers.
	Memory Usage: 16.5 MB, less than 69.57% of Python online submissions for Concatenation of Consecutive Binary Numbers.


### 原题链接

https://leetcode.com/problems/concatenation-of-consecutive-binary-numbers/


您的支持是我最大的动力
