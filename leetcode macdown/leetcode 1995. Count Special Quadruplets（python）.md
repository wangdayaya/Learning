leetcode  1995. Count Special Quadruplets（python）

### 描述

Given a 0-indexed integer array nums, return the number of distinct quadruplets (a, b, c, d) such that:

* nums[a] + nums[b] + nums[c] == nums[d], and
* a < b < c < d
 



Example 1:


	Input: nums = [1,2,3,6]
	Output: 1
	Explanation: The only quadruplet that satisfies the requirement is (0, 1, 2, 3) because 1 + 2 + 3 == 6.
	
Example 2:

	Input: nums = [3,3,6,4,5]
	Output: 0
	Explanation: There are no such quadruplets in [3,3,6,4,5].


Example 3:

	Input: nums = [1,1,1,3,5]
	Output: 4
	Explanation: The 4 quadruplets that satisfy the requirement are:
	- (0, 1, 2, 3): 1 + 1 + 1 == 3
	- (0, 1, 3, 4): 1 + 1 + 3 == 5
	- (0, 2, 3, 4): 1 + 1 + 3 == 5
	- (1, 2, 3, 4): 1 + 1 + 3 == 5




Note:

	4 <= nums.length <= 50
	1 <= nums[i] <= 100


### 解析


根据题意，给出一个从 0 开始索引的列表 nums ，题目要求我们返回不同四元组 ( a , b , c , d ) 的数量，使得：

* 	nums[a] + nums[b] + nums[c] == nums[d]
* 	a < b < c < d

最直接的办法就是使用暴力解法，思路如下：

* 初始化计数器 result 为 0
* 使用内置函数 combinations(nums, 4) 将所有四元组合都找出来，然后遍历所有的组合判断是否符合 cb[0] + cb[1] + cb[2] == cb[3] ，如果符合就计数器 result 加一
* 遍历结束返回 result 



### 解答
				
	class Solution(object):
	    def countQuadruplets(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        for cb in combinations(nums, 4):
	            if cb[0] + cb[1] + cb[2] == cb[3]:
	                result += 1
	        return result
	   
### 运行结果

	Runtime: 1780 ms, faster than 9.01% of Python online submissions for Count Special Quadruplets.
	Memory Usage: 13.3 MB, less than 74.59% of Python online submissions for Count Special Quadruplets.

### 解析


另外还可以用四层循环，寻找所有四个索引的组合来找出满足题意的四元组合，这也是一种暴力解法，只不过写法不同，运行的结果和上面的差不多。

### 解答

	class Solution(object):
	    def countQuadruplets(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = 0
	        for i in range(len(nums)-3):
	            for j in range(i+1, len(nums)-2):
	                for k in range(j+1, len(nums)-1):
	                    for l in range(k+1, len(nums)):
	                        if nums[i]+nums[j]+nums[k] == nums[l]:
	                            result += 1
	                            
	        return result
	            
	            

### 运行结果

	Runtime: 1588 ms, faster than 37.70% of Python online submissions for Count Special Quadruplets.
	Memory Usage: 13.4 MB, less than 74.59% of Python online submissions for Count Special Quadruplets.
	
	
### 解析

看了有论坛的大神用到了字典的解法很巧妙，主要是使用了 nums[a] + nums[b] == nums[d] - nums[c] 这一等式，将 nums[a] + nums[b]  保存入字典中，只需要遍历 c 和 d 的索引即可，时间复杂度可以降到 O(n^2) ，性能提升可是太大了！！

### 解答

	class Solution(object):
	    def countQuadruplets(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        n = len(nums)
	        res = 0
	        d = defaultdict(int)
	        d[nums[0] + nums[1]] = 1  
	        for i in range(2, n - 1):
	            for j in range(i + 1, n):
	                res += d[nums[j] - nums[i]]  
	            for j in range(i):
	                d[nums[i] + nums[j]] += 1 
	        return res

### 运行结果

	Runtime: 48 ms, faster than 99.18% of Python online submissions for Count Special Quadruplets.
	Memory Usage: 13.3 MB, less than 93.44% of Python online submissions for Count Special Quadruplets.

原题链接：https://leetcode.com/problems/count-special-quadruplets/



您的支持是我最大的动力
