leetcode  1925. Count Square Sum Triples（python）

### 描述

A square triple (a,b,c) is a triple where a, b, and c are integers and a^2 + b^2 = c^2.

Given an integer n, return the number of square triples such that 1 <= a, b, c <= n.



Example 1:

	Input: n = 5
	Output: 2
	Explanation: The square triples are (3,4,5) and (4,3,5).

	
Example 2:

	Input: n = 10
	Output: 4
	Explanation: The square triples are (3,4,5), (4,3,5), (6,8,10), and (8,6,10).



Note:

	1 <= n <= 250


### 解析


根据题意，就是找出在 [1,n] 范围内的三个数 (a,b,c) ，满足 a^2 + b^2 = c^2 ，这里有两点需要注意:

* 根据满足条件我们可以当作是在找直角三角形，也就是有一个隐形的条件就是 a+b>c 
* 在找出所有满足题意的 (a,b,c) 之后，因为 c 最大，所以它的位置是不能动的，但是可以转换 a 和 b 的位置，所以最后的三元组合数量是现有的三元组合的两倍

这种解法比较暴力，直接使用三次遍历，找出最后的所有三元组合。

### 解答
				

	class Solution(object):
	    def countTriples(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        result = 0
	        for i in range(1,n+1):
	            for j in range(i+1,n+1):
	                for k in range(j+1,n+1):
	                    if i+j>k and i*i + j*j == k*k:
	                        result += 1
	        return result*2
            	      
			
### 运行结果

	Runtime: 5060 ms, faster than 66.67% of Python online submissions for Count Square Sum Triples.
	Memory Usage: 13.4 MB, less than 100.00% of Python online submissions for Count Square Sum Triples.

### 解析

另外可以先将 1 到 n 的所有的平方结果存入集合 s 中，然后使用现有的组合排列的函数，将 s 中的值都进行排列组合成长度为 2 的组合 p ，然后遍历 p 中的每个组合的元素总和是否存在于 s 中，如果是则计数器 c 加一，遍历结束之后，将 c*2 返回即为结果。

### 解答

	class Solution(object):
	    def countTriples(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        s=set()
	        for i in range(1,n+1):
	            s.add(i*i)
	        p=combinations(s,2)
	        c=0
	        for i in p:
	            if(sum(list(i)) in s):
	                c=c+1
	        return c*2

### 运行结果

	Runtime: 252 ms, faster than 100.00% of Python online submissions for Count Square Sum Triples.
	Memory Usage: 13.4 MB, less than 100.00% of Python online submissions for Count Square Sum Triples.

### 解析

还有其他的解法，但是万变不离其宗，基本的原理都是一样的，不一样的只是代码形式而已，详细见官网解法。


原题链接：https://leetcode.com/problems/count-square-sum-triples/



您的支持是我最大的动力
