leetcode  1238. Circular Permutation in Binary Representation（python）

### 描述


Given 2 integers n and start. Your task is return any permutation p of (0,1,2.....,2^n -1) such that :

* p[0] = start
* p[i] and p[i+1] differ by only one bit in their binary representation.
* p[0] and p[2^n -1] must also differ by only one bit in their binary representation.



Example 1:

	Input: n = 2, start = 3
	Output: [3,2,0,1]
	Explanation: The binary representation of the permutation is (11,10,00,01). 
	All the adjacent element differ by one bit. Another valid permutation is [3,1,0,2]

	
Example 2:

	Input: n = 3, start = 2
	Output: [2,6,7,5,4,0,1,3]
	Explanation: The binary representation of the permutation is (010,110,111,101,100,000,001,011).



Note:

	1 <= n <= 16
	0 <= start < 2 ^ n


### 解析


根据题意，给定 2 个整数 n 和 start 。 题目要求是返回 ( 0 , 1 , 2 ..... , 2^n -1 ) 的任何排列 p 使得：

* p[0] = start
* p[i] 和 p[i+1] 在二进制表示中仅相差一位
* p[0] 和 p[2^n -1] 在其二进制表示中也必须仅相差一位

这道题其实就是给出了 grey code 的生成规则，可以使用 grey code 的规则直接来解题，grey code 的生成公式 i^(i>>1) ，生成整数序列列表 result ，然后找到 start 的位置索引 idx ，将 result[idx:] + result[:idx]  即可实现题目要求，返回答案即可。

### 解答
				

	class Solution(object):
	    def circularPermutation(self, n, start):
	        """
	        :type n: int
	        :type start: int
	        :rtype: List[int]
	        """
	        result = []
	        for i in range(1 << n):
	            result.append(i ^ (i >> 1))
	        idx = result.index(start)
	        return result[idx:] + result[:idx]  
	

            	      
			
### 运行结果

	Runtime: 196 ms, faster than 76.92% of Python online submissions for Circular Permutation in Binary Representation.
	Memory Usage: 23.1 MB, less than 46.15% of Python online submissions for Circular Permutation in Binary Representation.

### 解析

上面的方法比较取巧，知道公式的话直接就能解题，另外我们还能找规律解题。如例子二，我们可以从 0 还是生成符合题意的序列：

	000 初始化都为 0
	001 将 000 中最右边 0 变为 1 得到 001
	011 将 001 最右边 1 变为 0 得到 000 已经用过，所以只能变中间的 0 为 1 得到 011
	010 将 011 最右边 1 变为 0 得到 010
	110 将 010 最右边 0 变为 1 得到 011 已经用过，将中间的 1 变为 0 得到 000 已经用过，只能将最左边的 0 变为 1 得到 110
	111 将 110 最右边的 0 变为 1 得到 111
	101 将 111 最右边的 1 变为 0 得到 110 已经用过，将中间的 1 变为 0 得到 101
	100 将 101 最右边的 1 变为 0 得到 100 
	
规律已经能看出来了，那就是从 n 个 0 开始，从右往左依次将 0 和 1 互转 ，只要出现之前没有的序列就将其保留，并依次为基础序列，再从右往左依次进行 0 和 1 互转的操作，直到得到了 2^n 个序列列表 result 为止。这些序列肯定是满足题意要求的，然后再找到 start 开始的索引 idx ，将 result 变为 result[idx:]+result[:idx] 即可。

但是结果超时了，这种一个个找结果的方式对 2**n 个数来说计算量太大了。而且一般涉及到二进制的题用位运算效率为极大提升。
	

### 解答

	class Solution(object):
	    def circularPermutation(self, n, start):
	        """
	        :type n: int
	        :type start: int
	        :rtype: List[int]
	        """
	        result = []
	        L = ['0'] * n
	        while len(result) < 2**n:
	            if int(''.join(L), 2) not in result:
	                result.append(int(''.join(L), 2))
	            for i in range(n-1, -1, -1):
	                self.swap(L, i)
	                if int(''.join(L), 2) not in result:
	                    break
	                self.swap(L, i)
	        idx = result.index(start)
	        return result[idx:]+result[:idx]
	
	    def swap(self, L,i):
	        if L[i] == '1':
	            L[i] = '0'
	        else:
	            L[i] = '1'

### 运行结果

	Time Limit Exceeded


原题链接：https://leetcode.com/problems/circular-permutation-in-binary-representation/



您的支持是我最大的动力
