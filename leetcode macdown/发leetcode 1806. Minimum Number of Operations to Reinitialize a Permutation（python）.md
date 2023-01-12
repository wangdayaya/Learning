leetcode  1806. Minimum Number of Operations to Reinitialize a Permutation（python）

### 描述


You are given an even integer n​​​​​​. You initially have a permutation perm of size n​​ where perm[i] == i​ (0-indexed)​​​​.

In one operation, you will create a new array arr, and for each i:

* If i % 2 == 0, then arr[i] = perm[i / 2].
* If i % 2 == 1, then arr[i] = perm[n / 2 + (i - 1) / 2].

You will then assign arr​​​​ to perm.

Return the minimum non-zero number of operations you need to perform on perm to return the permutation to its initial value.




Example 1:

	Input: n = 2
	Output: 1
	Explanation: perm = [0,1] initially.
	After the 1st operation, perm = [0,1]
	So it takes only 1 operation.

	
Example 2:


	Input: n = 4
	Output: 2
	Explanation: perm = [0,1,2,3] initially.
	After the 1st operation, perm = [0,2,1,3]
	After the 2nd operation, perm = [0,1,2,3]
	So it takes only 2 operations.

Example 3:

	Input: n = 6
	Output: 4





Note:

	2 <= n <= 1000
	n​​​​​​ is even


### 解析


根据题意，就是给出了一个偶数 n ，最初可以形成一个列表 perm  ，以 0 为索引每个元素为 perm[i] == i​  , 我们进行一种特殊的操作，求至少做多少次这种操作，可以让 perm 又重新变回 perm 。这种操作就是：


* If i % 2 == 0, then arr[i] = perm[i / 2].
* If i % 2 == 1, then arr[i] = perm[n / 2 + (i - 1) / 2]

按照题意，思路比较简单，就是直接按照这种操作，不断进行，并使用计数器 count 计数，当 perm 重新变回 perm 时，返回 count 即可。

### 解答
				
	class Solution(object):
	    def reinitializePermutation(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        count = 0
	        perm = [x for x in range(n)]
	        result = [x for x in range(n)]
	        while True:
	            perm = [perm[i//2] if i % 2 ==0 else perm[n//2 + (i-1)//2] for i in range(n)]
	            count += 1
	            if perm==result:
	                return count
            	      
			
### 运行结果


	Runtime: 564 ms, faster than 38.89% of Python online submissions for Minimum Number of Operations to Reinitialize a Permutation.
	Memory Usage: 13.3 MB, less than 77.78% of Python online submissions for Minimum Number of Operations to Reinitialize a Permutation.
	
	
### 解析

另外，可以使用递归来解决 perm 的不断变化。


### 解答

	class Solution(object):
	    def __init__(self):
	        self.count = 0
	    def reinitializePermutation(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        perm = [x for x in range(n)]
	        result = [x for x in range(n)]
	        def f(perm):
	            perm = [perm[i//2] if i % 2 ==0 else perm[n//2 + (i-1)//2] for i in range(n)]
	            self.count += 1
	            if result == perm:
	                return
	            f(perm)
	        f(perm)
	        return self.count

### 运行结果

	Runtime: 600 ms, faster than 33.33% of Python online submissions for Minimum Number of Operations to Reinitialize a Permutation.
	Memory Usage: 21.4 MB, less than 11.11% of Python online submissions for Minimum Number of Operations to Reinitialize a Permutation.

原题链接：https://leetcode.com/problems/minimum-number-of-operations-to-reinitialize-a-permutation/



您的支持是我最大的动力
