leetcode  2044. Count Number of Maximum Bitwise-OR Subsets（python）

### 描述



Given an integer array nums, find the maximum possible bitwise OR of a subset of nums and return the number of different non-empty subsets with the maximum bitwise OR.

An array a is a subset of an array b if a can be obtained from b by deleting some (possibly zero) elements of b. Two subsets are considered different if the indices of the elements chosen are different.

The bitwise OR of an array a is equal to a[0] OR a[1] OR ... OR a\[a.length - 1] (0-indexed).

 

Example 1:

	Input: nums = [3,1]
	Output: 2
	Explanation: The maximum possible bitwise OR of a subset is 3. There are 2 subsets with a bitwise OR of 3:
	- [3]
	- [3,1]

	
Example 2:

	Input: nums = [2,2,2]
	Output: 7
	Explanation: All non-empty subsets of [2,2,2] have a bitwise OR of 2. There are 2^3 - 1 = 7 total subsets.


Example 3:


	Input: nums = [3,2,1,5]
	Output: 6
	Explanation: The maximum possible bitwise OR of a subset is 7. There are 6 subsets with a bitwise OR of 7:
	- [3,5]
	- [3,1,5]
	- [3,2,5]
	- [3,2,1,5]
	- [2,5]
	- [2,1,5]
	




Note:

	1 <= nums.length <= 16
	1 <= nums[i] <= 10^5



### 解析

根据题意，就是给出了一个正整数列表  nums ，找到 nums 中最大按位或的子集，符合条件的子集可能有多个，返回具有最大按位或的不同非空子集的数量。

如果可以通过删除列表 b 的 0 个或多个元素获得列表 a，则列表 a 是 b 的子集。如果所选元素的索引不同，则认为两个子集不同。

首先分别解释一下按位或和按位异或。
按位或，也就是本题中的 Bitwise-OR:
	
	按位或指的是参与运算的两个数分别对应的二进制位进行“或”的操作。只要对应的两个二进制位有一个为 1 时，结果位就为 1 。python中运算符为“|”

如：
	
	a=9 # 1001
	b=3 # 0011
	a|b  # 1011 ，十进制数字结果为 11 

按位异或：

	按位异或就是将参与运算的两个数对应的二进制位进行比较，如果一个位为 1 ，另一个位为 0 ，则结果为 1 ，否则结果位为 0 。python中运算符为 ”^“
	
如：
	
	a=9 # 1001
	b=3 # 0011
	a^b # 1010 ，十进制数字结果为 10

其实这道题也不难，就是靠对列表的排列组合，思路比较简单，我这里为了简单方便，直接用了 python 的内置排列组合函数 combinations ：

* 初始化字典 d ，一会用来保存不同的子集按位或结果及其个数
* 第一重循环 [1, len(nums)+1] 中的每个数字 i ，第二重循环 combinations(nums, i) 中的每个组合 c 
* 计算组合 c 的按位或结果 tmp ，如果 tmp 在字典 d 中，则 d[tmp] 加一，如果 tmp 不在 d 中，d[tmp] 设置为 1 
* 最后返回 d[max(d)] 即为最大按位或结果的不同子集个数

### 解答
				

	class Solution(object):
	    def countMaxOrSubsets(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        d = {}
	        for i in range(1, len(nums)+1):
	            for c in combinations(nums, i):
	                tmp = 0
	                for j in list(c):
	                    tmp |= j
	                if tmp in d:
	                    d[tmp] += 1
	                else:
	                    d[tmp] = 1
	        return d[max(d)]
            	      
			
### 运行结果

	Runtime: 1044 ms, faster than 44.83% of Python online submissions for Count Number of Maximum Bitwise-OR Subsets.
	Memory Usage: 13.5 MB, less than 60.34% of Python online submissions for Count Number of Maximum Bitwise-OR Subsets.


原题链接：https://leetcode.com/problems/count-number-of-maximum-bitwise-or-subsets/



您的支持是我最大的动力
