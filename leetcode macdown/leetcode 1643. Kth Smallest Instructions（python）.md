leetcode  1643. Kth Smallest Instructions（python）

### 描述


Bob is standing at cell (0, 0), and he wants to reach destination: (row, column). He can only travel right and down. You are going to help Bob by providing instructions for him to reach destination.

The instructions are represented as a string, where each character is either:

'H', meaning move horizontally (go right), or
'V', meaning move vertically (go down).
Multiple instructions will lead Bob to destination. For example, if destination is (2, 3), both "HHHVV" and "HVHVH" are valid instructions.

However, Bob is very picky. Bob has a lucky number k, and he wants the k<sup>th</sup> lexicographically smallest instructions that will lead him to destination. k is 1-indexed.

Given an integer array destination and an integer k, return the k<sup>th</sup> lexicographically smallest instructions that will take Bob to destination.

 


Example 1:


![](https://assets.leetcode.com/uploads/2020/10/12/ex1.png)
	
	Input: destination = [2,3], k = 1
	Output: "HHHVV"
	Explanation: All the instructions that reach (2, 3) in lexicographic order are as follows:
	["HHHVV", "HHVHV", "HHVVH", "HVHHV", "HVHVH", "HVVHH", "VHHHV", "VHHVH", "VHVHH", "VVHHH"].
	
Example 2:


![](https://assets.leetcode.com/uploads/2020/10/12/ex2.png)

	Input: destination = [2,3], k = 2
	Output: "HHVHV"

Example 3:


![](https://assets.leetcode.com/uploads/2020/10/12/ex3.png)

	Input: destination = [2,3], k = 3
	Output: "HHVVH"
	

Note:

	destination.length == 2
	1 <= row, column <= 15
	1 <= k <= nCr(row + column, row), where nCr(a, b) denotes a choose b​​​​​.


### 解析


根据题意，Bob 站在单元格 (0, 0) 的位置，他想到达目的地 (row, column) 。 他只能左右移动。 您将通过为 Bob 提供到达目的地的指示来帮助他。指令表示为一个字符串，其中每个字符的意思是：

* 'H'，意思是水平移动（向右移动）
* 'V'，意思是垂直移动（向下）

有多种字符串指令都能将引导 Bob 到达目的地。 例如，如果目的地是 (2, 3)，则 “HHHVV” 和 “HVHVH” 都是有效指令，这些字符串可以组成一个字符串集合，Bob 有一个幸运数字 k ，他想要将字符串集合进行字典排序，然后取第 k 个指令来引导他到达目的地。 k 是从 1 开始索引的。给定一个整数数组 destination 和一个整数 k ，返回将 Bob 带到目的地的第 k 个字典序的指令。

因为题目中给出了 destination ，所以 H 和 V 的个数是确定的，而且 H 在字典序上小于 V ，然后使用二分法解题，如例一所示 destination = [2,3] ，那么就说明有 3 个 H 和 2 个 V ：

* 按字典序集合中第 1 个指令的第一个位置肯定是 HXXXX （X 表示目前未知字符），后面的四个位置靠 2 个 H 和 2 个 V 组合而成有 6 种情况，所以如果 k 小于等于 6 那么肯定在 HXXXX 中，如果 k 大于 6 那么肯定在 VXXXX 
* 当第一位确定之后再用同样的方法确定第二位，如果第二个位置为 HHXXX , 后面靠 1 个 H 和 2 个 V 组成有 3 种情况，如果 k 小于等于 3 那么肯定在 HHXXX 中，如果 k 大于 3 那么肯定在 HVXXX 中
* 按照上面的规律进行排列不断找出指令即可

### 解答
				
	class Solution(object):
	    def kthSmallestPath(self, destination, k):
	        """
	        :type destination: List[int]
	        :type k: int
	        :rtype: str
	        """
	        H = destination[1]
	        V = destination[0]
	        result = ''
	        for i in range(destination[0]+destination[1]):
	            c = self.comb(H + V - 1, H - 1)
	            if H == 0:
	                result += 'V'
	            elif V == 0:
	                result += 'H'
	            elif k<=c:
	                result += 'H'
	                H -= 1 
	            else:
	                result += 'V'
	                k -= c
	                V -= 1   
	        return result
	    
	    def comb(self, n, m):
	        result = 1
	        for i in range(m):
	            result *= (n-i)
	            result //= (i+1)
	        return result

            	      
			
### 运行结果

	Runtime: 24 ms, faster than 62.50% of Python online submissions for Kth Smallest Instructions.
	Memory Usage: 13.4 MB, less than 37.50% of Python online submissions for Kth Smallest Instructions.


原题链接：https://leetcode.com/problems/kth-smallest-instructions/



您的支持是我最大的动力
