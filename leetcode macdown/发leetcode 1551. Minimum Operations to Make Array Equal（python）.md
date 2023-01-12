leetcode  1551. Minimum Operations to Make Array Equal（python）

### 描述

You have an array arr of length n where arr[i] = (2 * i) + 1 for all valid values of i (i.e. 0 <= i < n).

In one operation, you can select two indices x and y where 0 <= x, y < n and subtract 1 from arr[x] and add 1 to arr\[y] (i.e. perform arr[x] -=1 and arr[y] += 1). The goal is to make all the elements of the array equal. It is guaranteed that all the elements of the array can be made equal using some operations.

Given an integer n, the length of the array. Return the minimum number of operations needed to make all the elements of arr equal.



Example 1:

	
	Input: n = 3
	Output: 2
	Explanation: arr = [1, 3, 5]
	First operation choose x = 2 and y = 0, this leads arr to be [2, 3, 4]
	In the second operation choose x = 2 and y = 0 again, thus arr = [3, 3, 3].
	
Example 2:


	Input: n = 6
	Output: 9



Note:

	1 <= n <= 10^4


### 解析

根据题意，就是给出了一个整数 n ，构造列表 arr ，索引 i 位置上的整数为 arr[i] = (2 * i) + 1 ，然后执行一种操作，每次取不同的两个索引 x 和 y ，对  arr[x] 的值减一，对  arr[y] 的值加一，求一共执行多少次操作，能得到一个元素都相等的 arr 。其实这类题一般都是找规律的题，接下来我给大家找找看：

* 当 n = 1 ，arr=[1] ，因为元素都相等，所以不需要操作，直接返回 0 
* 当 n = 2 ，arr=[1,3]，只需要执行一次操作，即可得到 [2,2] ，直接返回 1
* 当 n = 3 ，arr=[1,3,5] ，执行操作过程如下，得到 [3,3,3] ，直接返回 2
	
		[2,3,4]
		[3,3,3]
* 当 n = 4 ，arr=[1,3,5,7] ，执行操作过程如下，得到 [4,4,4,4] ，直接返回 4
	
		[2,3,5,6]
		[3,3,5,5]
		[4,3,5,4]
		[4,4,4,4]

* 当 n = 5 ，arr=[1,3,5,7,9] ，执行操作过程如下，得到 [5,5,5,5] ，直接返回 6

		[2,3,5,7,8] 
		[3,3,5,7,7] 
		[4,3,5,7,6] 
		[5,3,5,7,5] 
		[5,4,5,6,5] 
		[5,5,5,5,5] 
		
* 当 n = 6 ，arr=[1,3,5,7,9,11] ，执行操作过程如下，得到 [6,6,6,6,6,6] ，直接返回 9

		[2,3,5,7,9,10] 
		[3,3,5,7,9,9] 
		[4,3,5,7,9,8] 
		[5,3,5,7,9,7] 
		[6,3,5,7,9,6] 
		[6,4,5,7,8,6] 
		[6,5,5,7,7,6] 
		[6,6,5,7,6,6] 
		[6,6,6,6,6,6] 

其实我们已经发现了规律：

* 当给出 n 的值，最后 arr 的所有相等的元素都为 n 
* 因为对前面一个元素加一，后面的元素就同时对应减一，所以只需要针对前面一半的元素，计算每个元素增加到 n 的次数的和即可得到答案

### 解答
				
	
	class Solution(object):
	    def minOperations(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        result = 0
	        for i in range(1,n+1,2):
	            result += n-i
	        return result
            	      
			
### 运行结果
	Runtime: 77 ms, faster than 30.60% of Python online submissions for Minimum Operations to Make Array Equal.
	Memory Usage: 13.9 MB, less than 17.16% of Python online submissions for Minimum Operations to Make Array Equal.


原题链接：https://leetcode.com/problems/minimum-operations-to-make-array-equal/



您的支持是我最大的动力
