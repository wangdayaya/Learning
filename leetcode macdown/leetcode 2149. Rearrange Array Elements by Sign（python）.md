leetcode  2149. Rearrange Array Elements by Sign（python）

### 前言



怎么说呢，因为参加双周赛太晚了，做完都夜里 12 点了，完事又没有睡意，所以搞的有点失眠，到了夜里三点半才睡着，所以早上起来参加  Weekly Contest 277 是有点云里雾里的，所幸题目不是很难，做出了三道，第四道没有头绪直接放弃了。这是 Weekly Contest 277 的第二题，难度 Medium ，考察的是对双指针的灵活使用，同样不是很难。



### 描述


You are given a 0-indexed integer array nums of even length consisting of an equal number of positive and negative integers.

You should rearrange the elements of nums such that the modified array follows the given conditions:

* Every consecutive pair of integers have opposite signs.
* For all integers with the same sign, the order in which they were present in nums is preserved.
* The rearranged array begins with a positive integer.

Return the modified array after rearranging the elements to satisfy the aforementioned conditions.

 


Example 1:


	Input: nums = [3,1,-2,-5,2,-4]
	Output: [3,-2,1,-5,2,-4]
	Explanation:
	The positive integers in nums are [3,1,2]. The negative integers are [-2,-5,-4].
	The only possible way to rearrange them such that they satisfy all conditions is [3,-2,1,-5,2,-4].
	Other ways such as [1,-2,2,-5,3,-4], [3,1,2,-2,-5,-4], [-2,3,-5,1,-4,2] are incorrect because they do not satisfy one or more conditions. 
	





Note:

	2 <= nums.length <= 2 * 10^5
	nums.length is even
	1 <= |nums[i]| <= 10^5
	nums consists of equal number of positive and negative integers.


### 解析

根据题意，给定一个以 0 为索引的整数数组 nums ，长度为偶数，由相等数量的正整数和负整数组成。题目要求我们重新排列 nums 的元素并返回，使修改后的数组遵循给定的条件：

* 每对连续的整数都有相反的符号。
* 对于所有具有相同符号的整数，它们在 nums 中出现的顺序被保留。
* 重新排列的数组以正整数开头。

题目应该是很简单的，题意也比较明晰，其实用最暴力的方法就是初始化两个列表 a 和 b ，a 存储 nums 中从左到右出现的正整数，b 存储 nums 中从左到右出现的负整数，因为两个列表的长度相等，我们只需要先从 a 中取第一个元素，b 中取第一个元素，然后 a 中取第二个元素， b 中取第二个元素，循环这个过程就可以得到结果，但是这个方法超时了，按道理这个算法的时间复杂度也就是 O(N) ，我比赛中没搞懂怎么回事，只能换个双指针了。


为了保证新的 nums 能有连续的相反符号的整数，并且他们各自的先后位置能保留，用我们最朴素的思路就是用两个指针 p 和 n ，一个找正整数，一个找负整数，从左到右遍历 nums 中的所有元素，先找正整数加入到新的列表中，再找负整数加入到列表中，再找下一个正整数，下一个负整数，重复这个过程，一直到给所有的元素都找完，新得到的列表 result 也就相当于重新进行了满足题意的排列。时间复杂度是 O(N)。

其实暴力解法就比双指针解法多了一次遍历的过程，难道就是因为这多了一次遍历就超时？搞不懂？

### 解答
				

	class Solution(object):
	    def rearrangeArray(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: List[int]
	        """
	        result = []
	        N = len(nums)
	        idx_p = 0
	        idx_n = 0
	        while len(result)<N:
	            while nums[idx_p]<0:
	                idx_p += 1
	            result.append(nums[idx_p])
	            idx_p += 1
	            while nums[idx_n]>0:
	                idx_n += 1
	            result.append(nums[idx_n])
	            idx_n += 1
	        return result
	            
            	      
			
### 运行结果


	133 / 133 test cases passed.
	Status: Accepted
	Runtime: 1512 ms
	Memory Usage: 46.5 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-277/problems/rearrange-array-elements-by-sign/


您的支持是我最大的动力
