leetcode  1310. XOR Queries of a Subarray（python）

### 描述


Given the array arr of positive integers and the array queries where queries[i] = [L<sub>i</sub>, R<sub>i</sub>], for each query i compute the XOR of elements from Li to Ri (that is, arr[L<sub>i</sub>] xor arr[L<sub>i+1</sub>] xor ... xor arr[R<sub>i</sub>] ). Return an array containing the result for the given queries.


Example 1:


	Input: arr = [1,3,4,8], queries = [[0,1],[1,2],[0,3],[3,3]]
	Output: [2,7,14,8] 
	Explanation: 
	The binary representation of the elements in the array are:
	1 = 0001 
	3 = 0011 
	4 = 0100 
	8 = 1000 
	The XOR values for queries are:
	[0,1] = 1 xor 3 = 2 
	[1,2] = 3 xor 4 = 7 
	[0,3] = 1 xor 3 xor 4 xor 8 = 14 
	[3,3] = 8
	
Example 2:

	Input: arr = [4,8,2,10], queries = [[2,3],[1,3],[0,0],[0,3]]
	Output: [8,0,4,4]



Note:

	1 <= arr.length <= 3 * 10^4
	1 <= arr[i] <= 10^9
	1 <= queries.length <= 3 * 10^4
	queries[i].length == 2
	0 <= queries[i][0] <= queries[i][1] < arr.length


### 解析


根据题意，就是给出了一个正整数列表 arr ，然后给出了查找索引范围列表 queries ，将每个索引查找范围内的 arr 元素都进行异或运算将结果加入到 reault 中并返回 result 。思路比较简单：

* 初始化结果列表 result 
* 就是直接对 queries 进行遍历
* 将 [L<sub>i</sub>, R<sub>i</sub>] 范围内的 arr 元素进行 ^ 运算，并将结果加入 result 
* 遍历结束返回 result 

从结果可以看出来超时了...啊哈，我就说这么简单的题肯定不是 Medium 难度的尿性。

### 解答
				
	class Solution(object):
	    def xorQueries(self, arr, queries):
	        """
	        :type arr: List[int]
	        :type queries: List[List[int]]
	        :rtype: List[int]
	        """
	        result = []
	        for query in queries:
	            tmp = arr[query[0]]
	            for i in range(query[0]+1, query[1]+1):
	                tmp ^= arr[i]
	            result.append(tmp)
	        return result

            	      
			
### 运行结果

	Time Limit Exceeded

### 解析

从题目的条件限制可以看出来，不管是 arr 的长度、还是 arr 中的元素、queries 的长度都是相当的大，所以直接像上面进行位运算肯定超时。

其实用到的原理还是 ^ 运算，但是这里有个小技巧，举个例子：

* 假如 arr = [1,2,3] 
* 计算后两个元素的异或结果 2^3=1 
* 但是我计算了 arr 前三个元素的异或结果，然后对第一个元素再做异或，即 (1^2^3)^1 也可以得到上面的结果

所以推广这个做法，我们求 query=[L,R] 范围内的结果，可以先求出 [0,L-1] 所有元素的异或得到 a ，再求出 [0,R] 所有元素的异或 b ，然后 a^b 即可得到题目要求的答案


### 解答

	class Solution(object):
	    def xorQueries(self, arr, queries):
	        """
	        :type arr: List[int]
	        :type queries: List[List[int]]
	        :rtype: List[int]
	        """
	        A = [0]
	        for a in arr:
	            A.append(A[-1] ^ a)
	        A.append(0)
	        result = []
	        for L, R in queries:
	            result.append(A[L]^A[R+1])
	        return result
	        
### 运行结果

	Runtime: 361 ms, faster than 31.82% of Python online submissions for XOR Queries of a Subarray.
	Memory Usage: 28.6 MB, less than 11.36% of Python online submissions for XOR Queries of a Subarray.


原题链接：https://leetcode.com/problems/xor-queries-of-a-subarray/



您的支持是我最大的动力
