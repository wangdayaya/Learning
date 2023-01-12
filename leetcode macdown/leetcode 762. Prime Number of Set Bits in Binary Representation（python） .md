leetcode  762. Prime Number of Set Bits in Binary Representation（python）

### 描述



Given two integers left and right, find the count of numbers in the range [left, right] (inclusive) having a prime number of set bits in their binary representation.

(Recall that the number of set bits an integer has is the number of 1s present when written in binary. For example, 21 written in binary is 10101 which has 3 set bits. Also, 1 is not a prime.)

Example 1:

	Input: left = 6, right = 10
	Output: 4
	Explanation:
	6 -> 110 (2 set bits, 2 is prime)
	7 -> 111 (3 set bits, 3 is prime)
	9 -> 1001 (2 set bits , 2 is prime)
	10->1010 (2 set bits , 2 is prime)

	
Example 2:

	Input: left = 10, right = 15
	Output: 5
	Explanation:
	10 -> 1010 (2 set bits, 2 is prime)
	11 -> 1011 (3 set bits, 3 is prime)
	12 -> 1100 (2 set bits, 2 is prime)
	13 -> 1101 (3 set bits, 3 is prime)
	14 -> 1110 (3 set bits, 3 is prime)
	15 -> 1111 (4 set bits, 4 is not prime)





Note:

	left, right will be integers left <= right in the range [1, 10^6].
	right - left will be at most 10000.



### 解析


根据题意，就是在 [left, right] 范围内的数字的二进制表达中包含的 1 的个数是质数的有多少个。自己定义了判断是否是质数的函数，然后遍历  [left, right] 的数的二进制包含的 1 的个数，遍历结果统计包含 1 的个数为质数的元素有多少个。

### 解答
				

	class Solution(object):
	    def countPrimeSetBits(self, left, right):
	        """
	        :type left: int
	        :type right: int
	        :rtype: int
	        """
	        def isPrime(n):
	            if n==1:
	                return False
	            if n==2:
	                return True
	            for i in range(2,n):
	                if n%i==0:
	                    return False
	            return True
	        result = 0
	        for i in range(left, right+1):
	            c = bin(i).count('1')
	            if isPrime(c):
	                result += 1
	        return result
            	      
			
### 运行结果
	
	Runtime: 420 ms, faster than 56.47% of Python online submissions for Prime Number of Set Bits in Binary Representation.
	Memory Usage: 13.6 MB, less than 58.82% of Python online submissions for Prime Number of Set Bits in Binary Representation.


### 解析

另外可以从题目中找出一些解题的窍门，因为我们的输入规定最大是 10^6 ，而最接近这个数字的并且满足题意的数是 983039 ，里面包含了 19 个 1 ，我们仅仅需要初始化一个变量 p 来记录 10^6 范围内的所有二进制中包含 1 的个数为质数的数，然后遍历  [left, right] 内的所有数字，判断每个元素转换成二进制后的字符串包含的 1 的个数是否在 p 中出现过，如果出现表示满足题意，计数器 result 加一，遍历结束之后，得到的 result 即为结果。


### 解答
	
	class Solution(object):
	    def countPrimeSetBits(self, left, right):
	        """
	        :type left: int
	        :type right: int
	        :rtype: int
	        """
	        p = {2, 3, 5, 7, 11, 13, 17, 19}
	        result = 0
	        for i in range(left, right+1):
	            if bin(i).count("1") in p:
	                result += 1
	        return result


### 运行结果

	Runtime: 144 ms, faster than 100.00% of Python online submissions for Prime Number of Set Bits in Binary Representation.
	Memory Usage: 13.7 MB, less than 25.00% of Python online submissions for Prime Number of Set Bits in Binary Representation.

	
原题链接：https://leetcode.com/problems/prime-number-of-set-bits-in-binary-representation/



您的支持是我最大的动力
