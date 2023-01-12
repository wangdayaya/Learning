leetcode  1952. Three Divisors（python）

### 描述
Given an integer n, return true if n has exactly three positive divisors. Otherwise, return false.

An integer m is a divisor of n if there exists an integer k such that n = k * m.




Example 1:


	Input: n = 2
	Output: false
	Explantion: 2 has only two divisors: 1 and 2.
	
Example 2:

	Input: n = 4
	Output: true
	Explantion: 4 has three divisors: 1, 2, and 4.


Note:


	1 <= n <= 10^4

### 解析

根据题意，就是给出了一个正整数 n ，判断这个 n 是否正好只有三个正整数除数，如例二：

	4 的除数正好有 1 、2 、4 三个正整数除数，所以返回 True
	
从题目中我们可以看出来，其实给出了 n 之后，除了 1 只有 1 个除数就是它本身之外，其他的数字的除数至少有两个除数已经确定，就是 1 和它本身 n ，所以我们的目标就是判断在 1 和 n 之间是否只有一个其他的除数。

* 当 n<=3 ，因为不满足题意，直接返回 False
* 因为一个数字 n 的最大的除数是 n//2 ，所以初始化一个 mid= n//2 ，初始化一个计数器 count 来记录出现的除数个数
* 从 2 开始一直到 mid 遍历每个数字，判断如果有数字能整除 n 则 count 加一，当 count>1 的时候，说明已经出现了 4 个除数，不符合题意直接返回 False 
* 遍历结束，如果 count 为 1 ，再加上 1 和 n 本身两个除数，表示正好有 3 个除数，返回 True ，否则不满足题意，返回 False 。


### 解答
				
	
	class Solution(object):
	    def isThree(self, n):
	        """
	        :type n: int
	        :rtype: bool
	        """
	        if n<=3:return False
	        mid = n//2
	        count = 0
	        for i in range(2, mid+1):
	            if n%i==0:
	                count += 1
	            if count>1:
	                return False
	        return count==1
            	      
			
### 运行结果

	Runtime: 20 ms, faster than 76.05% of Python online submissions for Three Divisors.
	Memory Usage: 13.6 MB, less than 60.78% of Python online submissions for Three Divisors.
	
### 解析

另外，深挖上面第一个思路，我们发现其实只有当质数的平方才能符题意，因为质数的平方肯定只有三个除数为 1 、质数本身、质数的平方 ，题目中规定了 n 最大为 10000 ，所以只有当 n 在 [4, 9, 25, 49, 121, 169, 289, 361, 529, 841, 961, 1369, 1681, 1849, 2209, 2809, 3481, 3721, 4489, 5041, 5329, 6241, 6889, 7921, 9409] 中才返回 True ，否则返回 False 。

### 解答

	class Solution(object):
	    def isThree(self, n):
	        """
	        :type n: int
	        :rtype: bool
	        """
	        return n in [4, 9, 25, 49, 121, 169, 289, 361, 529, 841, 961, 1369, 1681, 1849, 2209, 2809, 3481, 3721, 4489, 5041, 5329, 6241, 6889, 7921, 9409]
	


### 运行结果

	Runtime: 12 ms, faster than 99.10% of Python online submissions for Three Divisors.
	Memory Usage: 13.3 MB, less than 88.92% of Python online submissions for Three Divisors.

原题链接：https://leetcode.com/problems/three-divisors/



您的支持是我最大的动力
