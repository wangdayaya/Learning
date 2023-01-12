leetcode  119. Pascal's Triangle II（python）

### 描述

Given an integer rowIndex, return the rowIndex<sup>th</sup> (0-indexed) row of the Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:


![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fimages%2F20200410%2Fceb23eeae6c44e67bd9321d801792c98.gif&refer=http%3A%2F%2F5b0988e595225.cdn.sohucs.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1637720616&t=be0244f74d9e515015ee85264eb321db)




Example 1:

	Input: rowIndex = 3
	Output: [1,3,3,1]

	
Example 2:

	Input: rowIndex = 0
	Output: [1]


Example 3:

	Input: rowIndex = 1
	Output: [1,1]


Note:
	
	0 <= rowIndex <= 33


### 解析

根据题意，给出了一个整数 rowIndex ，要求我们返回杨辉三角中的第 rowIndex 行的内容。这道题和  [118. Pascal's Triangle](https://leetcode.com/problems/pascals-triangle/) 的考察点都是一样的，都是考察杨辉三角的基本原理，只不过 118 题中要返回前 n 行的所有内容，而本题只返回第 rowIndex 行的内容，大同小异，其实就是找规律题。

最简单的办法就是仿照 118 题中将杨辉三角的前 rowIndex 行内容都算出，最后只返回第 rowIndex 行的内容即可。但是很明显这种速度太慢了。


### 解答
				


	class Solution(object):
	    def getRow(self, rowIndex):
	        """
	        :type rowIndex: int
	        :rtype: List[int]
	        """
	        if rowIndex == 0: return [1]
	        if rowIndex == 1: return [1,1]
	        result = [[0],[1,1]]
	        for i in range(2, rowIndex+1):
	            tmp = [1]
	            for j in range(1, i):
	                tmp.append(result[-1][j-1]+result[-1][j])
	            tmp.append(1)
	            result.append(tmp)
	        return result[-1]
	        
            	      
			
### 运行结果

	        
    Runtime: 35 ms, faster than 10.55% of Python online submissions for Pascal's Triangle II.
	Memory Usage: 13.1 MB, less than 99.32% of Python online submissions for Pascal's Triangle II.	
	            
### 解析

可以直接在一个列表上进行所有行的杨辉三角的加法操作，这样可以提升运算效率。

### 解答

	class Solution(object):
	    def getRow(self, rowIndex):
	        """
	        :type rowIndex: int
	        :rtype: List[int]
	        """
	        result = [1]*(rowIndex + 1)
	        for i in range(2,rowIndex+1):
	            for j in range(i-1,0,-1):
	                result[j] += result[j-1]
	        return result
	            

### 运行结果
	
	Runtime: 8 ms, faster than 99.57% of Python online submissions for Pascal's Triangle II.
	Memory Usage: 13.5 MB, less than 39.40% of Python online submissions for Pascal's Triangle II.


### 相关题目


* [118. Pascal's Triangle](https://leetcode.com/problems/pascals-triangle/)

原题链接：https://leetcode.com/problems/pascals-triangle-ii/



您的支持是我最大的动力
