leetcode  2293. Min Max Game（python）



### 描述

You are given a 0-indexed integer array nums whose length is a power of 2.

Apply the following algorithm on nums:

* Let n be the length of nums. If n == 1, end the process. Otherwise, create a new 0-indexed integer array newNums of length n / 2.
* For every even index i where 0 <= i < n / 2, assign the value of newNums[i] as min(nums[2 * i], nums[2 * i + 1]).
* For every odd index i where 0 <= i < n / 2, assign the value of newNums[i] as max(nums[2 * i], nums[2 * i + 1]).
* Replace the array nums with newNums.
* Repeat the entire process starting from step 1.

Return the last number that remains in nums after applying the algorithm.



Example 1:


![](https://assets.leetcode.com/uploads/2022/04/13/example1drawio-1.png)

	Input: nums = [1,3,5,2,4,8,2,2]
	Output: 1
	Explanation: The following arrays are the results of applying the algorithm repeatedly.
	First: nums = [1,5,4,2]
	Second: nums = [1,4]
	Third: nums = [1]
	1 is the last remaining number, so we return 1.
	
Example 2:


	Input: nums = [3]
	Output: 3
	Explanation: 3 is already the last remaining number, so we return 3.



Note:



	1 <= nums.length <= 1024
	1 <= nums[i] <= 10^9
	nums.length is a power of 2.

	

### 解析

根据题意，给定一个 0 开始索引的数组 nums ，长度为 2 的倍数，让我们按照特定的算法来改变 nums ：

* 假设 n 为 nums 的长度。 如果 n == 1 ，则结束该过程。 否则，创建一个长度为 n / 2 的新的 0 索引整数数组 newNums。
* 对于每个 0 <= i < n / 2 的偶数索引 i ，将 newNums[i] 的值设置为 min(nums[2 * i], nums[2 * i + 1]) 。
* 对于 0 <= i < n / 2 的每个奇数索引 i ，将 newNums[i] 的值设置为 max(nums[2 * i], nums[2 * i + 1])。
* 将数组 nums 替换为 newNums。
* 从步骤 1 开始重复整个过程。

返回执行算法后保留在 nums 中的最后一个数字。

我们直接按照题意，进行数组的修改即可。时间复杂度为 O(logN) ，空间复杂度为 O(N) 。


### 解答
				

	class Solution(object):
	    def minMaxGame(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        if len(nums) == 1: return nums[0]
	        new = []
	        while len(new) != 1:
	            new = [0] * (len(nums) // 2)
	            for i in range(0, len(nums), 2):
	                if (i // 2) % 2 == 0:
	                    tmp = nums[i:i + 2]
	                    new[i//2] = min(tmp)
	                else:
	                    tmp = nums[i:i + 2]
	                    new[i//2] = max(tmp)
	            nums = new
	        return new[0]
            	      
			
### 运行结果


	
	96 / 96 test cases passed.
	Status: Accepted
	Runtime: 41 ms
	Memory Usage: 13.7 MB

### 原题链接

	https://leetcode.com/contest/weekly-contest-296/problems/min-max-game/

您的支持是我最大的动力
