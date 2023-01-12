leetcode  2148. Count Elements With Strictly Smaller and Greater Elements（python）

### 前言



上周六晚上刚参加了 Biweekly Contest 70  ，然后上周日上午又紧接着参加了 Weekly Contest 277 ，真的是充实的周末【苦瓜脸】，很明显这次的单周赛比双周赛的难度小了很多，因为榜单第一名做完四道题只用了四分钟，而且榜单前两名都是中国选手，这可是凭实力霸榜，点赞。我只做出来三道题，真的是惭愧。这是  Weekly Contest 277 的第一题，难度 Eazy ，考察的就是基本的排序，只要理清思路就基本上能做出来，第一题都做不出来那可真的是完犊子了。




### 描述


Given an integer array nums, return the number of elements that have both a strictly smaller and a strictly greater element appear in nums.


Example 1:


	Input: nums = [11,7,2,15]
	Output: 2
	Explanation: The element 7 has the element 2 strictly smaller than it and the element 11 strictly greater than it.
	Element 11 has element 7 strictly smaller than it and element 15 strictly greater than it.
	In total there are 2 elements having both a strictly smaller and a strictly greater element appear in nums.
	
Example 2:


	Input: nums = [-3,3,3,90]
	Output: 2
	Explanation: The element 3 has the element -3 strictly smaller than it and the element 90 strictly greater than it.
	Since there are two elements with the value 3, in total there are 2 elements having both a strictly smaller and a strictly greater element appear in nums.



Note:


	1 <= nums.length <= 100
	-10^5 <= nums[i] <= 10^5

### 解析


根据题意，给定一个整数数组 nums，返回在 nums 中出现的同时具有严格更小和严格更大元素的元素的数量。

其实题干很简单，题意也很明白，只要某个元素在 nums 中有比其小的元素存在，并且有比其大的元素存在，那么就将其加入到计数器 result 中。所以解法如下：

* 初始化 result 为 0 
* 因为 nums 的长度大于等于 1 ，所以直接进行排序即可，便于我们之后进行的判断
* 经过排序之后，因为第一个元素是最小的元素，肯定没有比其更小的元素，所以第一个元素肯定不是符合题意的元素
* 最后一个元素是最大的元素，肯定没有比其更大的元素，所以最后一个元素肯定也不是符合题意的元素
* 我们从第二个元素开始遍历，只要元素大于 nums[0] ，小于 nums[N-1] （N 为 nums 的长度），那么我们就判定这个元素符合题意，计数器 result 加一
* 遍历结束直接返回 reuslt 即可

### 解答
				
	
	class Solution(object):
	    def countElements(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        nums.sort()
	        N = len(nums)
	        result = 0
	        for i in range(1, N):
	            if nums[i]>nums[0] and nums[i]<nums[N-1]:
	                result += 1
	        return result
	                
	            
            	      
			
### 运行结果

	
	127 / 127 test cases passed.
	Status: Accepted
	Runtime: 38 ms
	Memory Usage: 13.4 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-277/problems/count-elements-with-strictly-smaller-and-greater-elements/



您的支持是我最大的动力
