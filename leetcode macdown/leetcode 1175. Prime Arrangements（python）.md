leetcode  1175. Prime Arrangements（python）

### 描述

Return the number of permutations of 1 to n so that prime numbers are at prime indices (1-indexed.)

(Recall that an integer is prime if and only if it is greater than 1, and cannot be written as a product of two positive integers both smaller than it.)

Since the answer may be large, return the answer modulo 10^9 + 7.





Example 1:

	Input: n = 5
	Output: 12
	Explanation: For example [1,2,5,4,3] is a valid permutation, but [5,2,3,4,1] is not because the prime number 5 is at index 1.


	
Example 2:


	Input: n = 100
	Output: 682289015


Note:

	1 <= n <= 100


### 解析

根据题意，就是给出 n 个从 1 到 n 的数字，将其得到其不同排列的样式数量，必须满足将质数放到质数索引的位置上（索引从 1 开始），所以先找出 n 个数中的质数的数量 num_prime ，非质数数量为 n-num_prime ，然后对质数和非质数数量各自做阶乘，并将两个阶乘结果相乘后对 10**9+7 取模。


### 解答
				
	class Solution(object):
	    def numPrimeArrangements(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        num_prime = 0
	        for i in range(2, n+1):
	            for j in range(2, i):
	                if i%j==0:
	                    break
	            else:
	                num_prime+=1
	        r = 1
	        for i in range(num_prime,0,-1):
	            r *= i
	        for i in range(n-num_prime,0,-1):
	            r *= i
	        return r%(10**9+7)

            	      
			
### 运行结果


	Runtime: 24 ms, faster than 40.91% of Python online submissions for Prime Arrangements.
	Memory Usage: 13.3 MB, less than 93.18% of Python online submissions for Prime Arrangements.
	
	
### 解析

思路和上面类似，只不过题目限制了 n 在 100 以内，所以可以取巧先将 100 以内的质数都列出来，然后找出 n 以内质数的数量 num_primes ，然后使用内置函数求 num_primes 和 n-num_primes 的阶乘，并将两者阶乘相乘对 10 ** 9 + 7 取模。


### 解答
				
					
	class Solution(object):
	    def numPrimeArrangements(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37, 41, 43, 47, 53, 59, 61, 67, 71, 73, 79, 83, 89, 97]
	        num_primes = len([x for x in primes if x <= n])  # cleaned up per comment
	        return (factorial(num_primes) * factorial(n - num_primes)) % (10 ** 9 + 7)				
            	      
			
### 运行结果	
	
	Runtime: 12 ms, faster than 100.00% of Python online submissions for Prime Arrangements.
	Memory Usage: 13.4 MB, less than 40.91% of Python online submissions for Prime Arrangements.

原题链接：https://leetcode.com/problems/prime-arrangements/



您的支持是我最大的动力
