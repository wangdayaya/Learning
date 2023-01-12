leetcode  2011. Final Value of Variable After Performing Operations（python）

### 描述

There is a programming language with only four operations and one variable X:

* ++X and X++ increments the value of the variable X by 1.
* --X and X-- decrements the value of the variable X by 1.

Initially, the value of X is 0.

Given an array of strings operations containing a list of operations, return the final value of X after performing all the operations.

 



Example 1:

	Input: operations = ["--X","X++","X++"]
	Output: 1
	Explanation: The operations are performed as follows:
	Initially, X = 0.
	--X: X is decremented by 1, X =  0 - 1 = -1.
	X++: X is incremented by 1, X = -1 + 1 =  0.
	X++: X is incremented by 1, X =  0 + 1 =  1.

	
Example 2:


	Input: operations = ["++X","++X","X++"]
	Output: 3
	Explanation: The operations are performed as follows:
	Initially, X = 0.
	++X: X is incremented by 1, X = 0 + 1 = 1.
	++X: X is incremented by 1, X = 1 + 1 = 2.
	X++: X is incremented by 1, X = 2 + 1 = 3.

Example 3:

	
	Input: operations = ["X++","++X","--X","X--"]
	Output: 0
	Explanation: The operations are performed as follows:
	Initially, X = 0.
	X++: X is incremented by 1, X = 0 + 1 = 1.
	++X: X is incremented by 1, X = 1 + 1 = 2.
	--X: X is decremented by 1, X = 2 - 1 = 1.
	X--: X is decremented by 1, X = 1 - 1 = 0.
	


Note:

	1 <= operations.length <= 100
	operations[i] will be either "++X", "X++", "--X", or "X--".


### 解析


根据题意，就是给出来某种编程语言的的四种操作和一个变量 X ：

* 如果操作为 ++X 或者 X++ 表示对 X 做加一操作
* 如果操作为 --X 或者 X-- 表示对 X 做减一操作

我们初始化变量 X 为 0 ，问在执行了一系列的 operations 之后得到的最终值是多少。

其实思路很简单，就是遍历 operations 中的每一个操作 op ，如果字符串 op 中包含 - 表示的是减一操作，如果字符串 op 中包含 + 表示的是加一操作，遍历结束得到的结果就是答案。

### 解答
				
	class Solution(object):
	    def finalValueAfterOperations(self, operations):
	        """
	        :type operations: List[str]
	        :rtype: int
	        """
	        result = 0
	        for op in operations:
	            if '-' in op:
	                result -= 1
	            elif '+' in op:
	                result += 1
	        return result

            	      
			
### 运行结果


	Runtime: 24 ms, faster than 99.62% of Python online submissions for Final Value of Variable After Performing Operations.
	Memory Usage: 13.6 MB, less than 13.41% of Python online submissions for Final Value of Variable After Performing Operations.

### 解析

另外我们可以用 python 的内置函数 sum ，代码简洁易懂。

sum(iterable[, start]) 函数对序列进行求和计算，传入的对象可以是列表、元组、集合等可迭代对象。例如：

	>>>sum([0,1,2])  
	3  

### 解答

	class Solution(object):
	    def finalValueAfterOperations(self, operations):
	        """
	        :type operations: List[str]
	        :rtype: int
	        """
	        return sum(1 if '+' in o else -1 for o in operations)

### 运行结果

	Runtime: 28 ms, faster than 99.23% of Python online submissions for Final Value of Variable After Performing Operations.
	Memory Usage: 13.3 MB, less than 86.21% of Python online submissions for Final Value of Variable After Performing Operations.

原题链接：https://leetcode.com/problems/final-value-of-variable-after-performing-operations/



您的支持是我最大的动力
