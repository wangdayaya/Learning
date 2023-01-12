leetcode  31. Next Permutation（python）

### 描述

Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such an arrangement is not possible, it must rearrange it as the lowest possible order (i.e., sorted in ascending order).

The replacement must be in place and use only constant extra memory.




Example 1:

	Input: nums = [1,2,3]
	Output: [1,3,2]

	
Example 2:

	Input: nums = [3,2,1]
	Output: [1,2,3]


Example 3:
	
	Input: nums = [1,1,5]
	Output: [1,5,1]

	
Example 4:

	Input: nums = [1]
	Output: [1]


Note:

	1 <= nums.length <= 100
	0 <= nums[i] <= 100


### 解析


根据题意，就是给出了一个包含了整数数字的列表 nums ，这个列表的元素从左到右可以形成一个数字，我们要实现一个功能，将 nums 中的元素进行重新排列得到的新列表形成的数字是刚刚好比初始的数字大。

如果这样的排列是不可能找到的，则必须将其重新排列为升序顺序。而且题目要求我们不能使用额外的空间，也就是说只能在原来的 nums 列表上进行操作。

其实这道题是可以用暴力解法求解的，但是太耗时间了，还是不推荐。其实这是一个找规律题，假如我们现在有一个列表 [4,7,5,3] ，下一个比他大的列表应该是 [5, 3, 4, 7]，基本思路就是遵循最小的排列就是列表升序结果，最大的排列就是列表降序结果，过程如下：

* 从左往右遍历，最后一位是 [3] ，单个数字不进行操作
* 后两位是 [5，3] ，5 大于 3 ，这个子列表已经组成的是最大数，无法再找出更大的数，所以也不进行操作
* 后三位是 [7，5，3]，同上一样，也不进行操作
* 后四位是 [4，7，5，3] ，此时 4 小于 7 且小于 5 ，而 [4，7，5，3] 的下一个比他大的排列应该是比 [4，7，5，3] 大的结果集合中的最小的一个，所以应该将 4 和 5 进行交换得到 [5，7，4，3] ，此时 5 位于第一位，后面的子列表应该让其最小化，即 [3，4，7]，拼接起来最后的结果为 [5，3，4，7]


总结下来就是步骤就是：

* 从右到左找到第一个降序的数字索引 i ，例子中的数字是 4
* 从右到左找到第一个比 nums[i] 大的数字的索引 j ，例子中的数字是 5
* 交换索引 i 和 j 的元素
* 将 nums[i+1:] 的元素进行升序排列


### 解答
				
	class Solution(object):
	    def nextPermutation(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: None Do not return anything, modify nums in-place instead.
	        """
	        i = len(nums) - 2
	        while i>=0 and nums[i] >= nums[i+1]:
	            i -= 1
	        if i>=0: 
	            j = len(nums)-1
	            while (nums[j]<=nums[i]):
	                j -= 1
	            self.swap(nums, i, j)
	        self.reverse(nums, i+1)
	    
	    def reverse(self, nums, start):
	        i = start
	        j = len(nums)-1
	        while (i<j):
	            self.swap(nums, i, j)
	            i+=1
	            j-=1
	            
	    def swap(self, nums, i, j):
	        nums[i], nums[j] = nums[j], nums[i]
	        
	        
	    

            	      
			
### 运行结果

	
	
	Runtime: 32 ms, faster than 58.10% of Python online submissions for Next Permutation.
	Memory Usage: 13.3 MB, less than 71.68% of Python online submissions for Next Permutation.
	
### 解析


上面的解法过程应该很详细了，不懂的多看几次就懂了，只是代码部分自己实现了几个函数，显得比较繁琐，我们可以简化一下，使用 python 的内置函数来解题会更好一点。


### 解答
	
	class Solution(object):
	    def nextPermutation(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: None Do not return anything, modify nums in-place instead.
	        """
	        i = len(nums)-1
	        while i>0:
	            if nums[i-1]<nums[i]:
	                break
	            i = i-1
	        i = i-1
	        j = len(nums)-1
	        while j>i:
	            if nums[j]>nums[i]:
	                break
	            j=j-1
	        nums[i],nums[j]=nums[j],nums[i]  
	        nums[i+1:]=sorted(nums[i+1:]) 

### 运行结果
	
	Runtime: 28 ms, faster than 77.80% of Python online submissions for Next Permutation.
	Memory Usage: 13.2 MB, less than 92.31% of Python online submissions for Next Permutation.

原题链接：https://leetcode.com/problems/next-permutation/



您的支持是我最大的动力
