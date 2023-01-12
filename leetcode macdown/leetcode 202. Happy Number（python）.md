leetcode  202. Happy Number（python）

### 描述

Write an algorithm to determine if a number n is happy.

A happy number is a number defined by the following process:

* Starting with any positive integer, replace the number by the sum of the squares of its digits.
* Repeat the process until the number equals 1 (where it will stay), or it loops endlessly in a cycle which does not include 1.
* Those numbers for which this process ends in 1 are happy.

Return true if n is a happy number, and false if not.



Example 1:

	Input: n = 19
	Output: true
	Explanation:
	12 + 92 = 82
	82 + 22 = 68
	62 + 82 = 100
	12 + 02 + 02 = 1

	
Example 2:

	Input: n = 2
	Output: false




Note:

	1 <= n <= 2^31 - 1


### 解析


根据题意，要求我们写一个算法检测给出的整数 n 是否是快乐的数字。

快乐的数字是由以下过程定义的数字：

* 起始的数字是任何一个正整数，用其各个单个数字的平方的总和替换该数字。
* 重复上面这个过程直到数字最终等于 1 ，或者它在一个不包括 1 的循环中进行无限循环。
* 在这个过程以 1 结束的 n 是快乐的。

如果 n 是一个快乐的数字，则返回 true，否则返回 false。

思路比较简单，循环终止条件有两个：经过计算数字 n 中每个单数字的平方和为 1 ，或经过计算得到数字 n 的平方和在之前的循环中出现过，但是只有最后以 1 终止的才是快乐数字。

判断平方和是否为 1 只需要每次判断即可；而判断平方和是否出现，则只需要维持一个集合，检查当前平方和是否在集合中且不等于 1 ，如果不满足条件在则终止循环，否则则将此平方和放到集合中继续循环。最终在结束之后判断 n 是否等于 1 即可。

### 解答
				

	class Solution(object):
	    def isHappy(self, n):
	        """
	        :type n: int
	        :rtype: bool
	        """
	        def getSUM(n):
	            result = 0
	            while n>0:
	                result += (n%10)**2
	                n //= 10
	            return result
	        s = set()
	        while n!=1 and n not in s:
	            s.add(n)
	            n = getSUM(n)
	        return n==1
            	      
			
### 运行结果


	Runtime: 20 ms, faster than 81.58% of Python online submissions for Happy Number.
	Memory Usage: 13.4 MB, less than 67.99% of Python online submissions for Happy Number.


原题链接：https://leetcode.com/problems/happy-number



您的支持是我最大的动力
