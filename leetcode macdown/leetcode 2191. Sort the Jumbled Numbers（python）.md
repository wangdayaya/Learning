leetcode  2191. Sort the Jumbled Numbers（python）



### 前言

这是 Biweekly Contest 73 的第二题，考察的是对列表的排序，难度 Medium ，使用 python 的内置函数 sorted 直接解题即可。

### 描述


You are given a 0-indexed integer array mapping which represents the mapping rule of a shuffled decimal system. mapping[i] = j means digit i should be mapped to digit j in this system.The mapped value of an integer is the new integer obtained by replacing each occurrence of digit i in the integer with mapping[i] for all 0 <= i <= 9.You are also given another integer array nums. Return the array nums sorted in non-decreasing order based on the mapped values of its elements.

Notes:

* Elements with the same mapped values should appear in the same relative order as in the input.
* The elements of nums should only be sorted based on their mapped values and not be replaced by them.


Example 1:


	Input: mapping = [8,9,4,0,2,1,3,5,7,6], nums = [991,338,38]
	Output: [338,38,991]
	Explanation: 
	Map the number 991 as follows:
	1. mapping[9] = 6, so all occurrences of the digit 9 will become 6.
	2. mapping[1] = 9, so all occurrences of the digit 1 will become 9.
	Therefore, the mapped value of 991 is 669.
	338 maps to 007, or 7 after removing the leading zeros.
	38 maps to 07, which is also 7 after removing leading zeros.
	Since 338 and 38 share the same mapped value, they should remain in the same relative order, so 338 comes before 38.
	Thus, the sorted array is [338,38,991].
	



Note:

	mapping.length == 10
	0 <= mapping[i] <= 9
	All the values of mapping[i] are unique.
	1 <= nums.length <= 3 * 10^4
	0 <= nums[i] < 10^9


### 解析

根据题意，就是给出了一个 0 索引的整数列表 mapping ，mapping[i] = j 表示的是数字 i 可以映射变成数字 j ，这里的映射范围已经定好了，就是对 0 到 9 之间的数字进行处理。

又给了我们一个整数列表 nums ，让我们对里面的整数进行非降序排序，排序是基于每个元素 mapping 变化之后的大小进行的，同时还有以下两点要注意：

* 如果映射之后的值相同，则要保证他们的相对位置和原来一样
* 我们是对 nums 中的元素进行排序，并不是将 nums 中的元素通过 mapping 进行变化然后再排序



其实这道题很明显就是考察的是一个排序的基本操作，我们可以使用 python 内置的sorted 函数对 nums 进行排序，然后自定义一个函数 custom 计算每个元素经过映射后的值， nums 可以依据这个函数来进行非降序排序。思路比较清晰，代码也很简单。

因为 nums 中元素最多有 3 \* 10^4 ，每个元素最大为 10 位数字，我们对所有元素进行 mapping 操作计算的时间复杂度为 O(K \* N) ，然后经过排序时间复杂度为 O(NlogN) ，所以总的时间复杂度为  O(K \* N) + O(NlogN) ，空间复杂度 O(N) 。

### 解答
				

	class Solution(object):
	    def sortJumbled(self, mapping, nums):
	        """
	        :type mapping: List[int]
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        def custom(x):
	            if x==0: return mapping[0]
	            result = 0
	            p = 0
	            while x > 0:
	                t = x % 10
	                result += mapping[t] * pow(10, p)
	                x //= 10
	                p += 1
	            return result
	        return sorted(nums, key=custom)  	      
			
### 运行结果


	
	66 / 66 test cases passed.
	Status: Accepted
	Runtime: 1985 ms
	Memory Usage: 19.8 MB

### 原题链接



https://leetcode.com/contest/biweekly-contest-73/problems/sort-the-jumbled-numbers/


您的支持是我最大的动力
