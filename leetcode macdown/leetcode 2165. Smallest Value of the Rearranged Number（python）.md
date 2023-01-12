leetcode 2165. Smallest Value of the Rearranged Number （python）

### 前言

这是 Weekly Contest 279 比赛的第二题，难度 Medium ，考察的是对题目的理解和数组的操作，很简单。


### 描述

You are given an integer num. Rearrange the digits of num such that its value is minimized and it does not contain any leading zeros.

Return the rearranged number with minimal value.

Note that the sign of the number does not change after rearranging the digits.



Example 1:

	Input: num = 310
	Output: 103
	Explanation: The possible arrangements for the digits of 310 are 013, 031, 103, 130, 301, 310. 
	The arrangement with the smallest value that does not contain any leading zeros is 103.

	




Note:

* -10^15 <= num <= 10^15


### 解析


根据题意，给一个整数 num。 重新排列 num 的数字，使其值最小化并且不包含任何前置零。返回具有最小值的重新排列的数字。需要注意的是，数字有正负，重新排列数字后正负要保持不变。

这个题虽然是个 Medium 难度，但是其实根据题意我们直接写代码就可以了，如果 num 是正数：

* 将 num 中所有的数字都放入一个列表 L 中，并按照升序排序
* 因为是正数，想要改变之后值最小，要让这些数字组成的值最小即可，但是要防止第一个数字是 0 ，所以我们要遍历 L 中的元素，从左往右找出第一个不为 0 的数字，然后让其和索引为 0 的元素进行交换，最后将列表 L 中的元素合并成字符串，在转换为整数返回即可

如果 num 是负数：

* 将 num 中所有的数字都放入一个列表 L 中，并按照降序排序
* 因为是负数，想要改变之后值最小，要让这些数字组成的值最大即可，而我们的 L 已经是按照降序进行了排列，所以直接将 L 中的元素进行合并，转换成负整数返回即可

时间复杂度是 O(N) ，空间复杂度是 O(N) 。当然了这个代码肯定是有些冗余的，其实精简之后只需要不到十行代码即可解决。

### 解答
				
	class Solution(object):
	    def smallestNumber(self, num):
	        """
	        :type num: int
	        :rtype: int
	        """
	        # 负数，要保证数字部分最大
	        if num < 0:
	            L = sorted(list(str(num)[1:]), reverse=True)
	            return -int(''.join(L))
	        # 0，保持不变
	        elif num == 0:
	            return 0
	        # 正数，要保证数字部分最小
	        else:
	            L = sorted(list(str(num)))
	            idx = 0
	            while idx<len(L):
	                if L[idx]!='0':
	                    break
	                idx += 1
	            L[idx], L[0] = L[0], L[idx]
	            return int(''.join(L))
	        
	        
	        
	        
### 运行结果

	

	413 / 413 test cases passed.
	Status: Accepted
	Runtime: 18 ms
	Memory Usage: 13.2 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-279/problems/smallest-value-of-the-rearranged-number/


您的支持是我最大的动力
