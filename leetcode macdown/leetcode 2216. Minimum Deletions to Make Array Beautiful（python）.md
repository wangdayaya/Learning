leetcode  2216. Minimum Deletions to Make Array Beautiful（python）




### 描述


You are given a 0-indexed integer array nums. The array nums is beautiful if:

* nums.length is even.
* nums[i] != nums[i + 1] for all i % 2 == 0.

Note that an empty array is considered beautiful.

You can delete any number of elements from nums. When you delete an element, all the elements to the right of the deleted element will be shifted one unit to the left to fill the gap created and all the elements to the left of the deleted element will remain unchanged.

Return the minimum number of elements to delete from nums to make it beautiful.


Example 1:


	Input: nums = [1,1,2,3,5]
	Output: 1
	Explanation: You can delete either nums[0] or nums[1] to make nums = [1,2,3,5] which is beautiful. It can be proven you need at least 1 deletion to make nums beautiful.
	




Note:

1 <= nums.length <= 10^5
0 <= nums[i] <= 10^5


### 解析


根据题意，给你一个 0 索引的整数数组 nums。 如果满足以下条件，则数组 nums 是 beautiful ，空数组也被认为是 beautiful：

* nums.length 是偶数。
* nums[i] != nums[i + 1] 对于所有 i % 2 == 0。

我们可以从 nums 中删除任意数量的元素。 当删除一个元素时，被删除元素右侧的所有元素都将向左移动一个单位以填补创建的空白，而被删除元素左侧的所有元素将保持不变。返回要从 nums 中删除的最小元素数以使其成为 beautiful 的数组。

这道题明显就是贪心算法，用我们最朴素的想法肯定就是从左到右遍历 nums ，如果索引 i 为偶数的时候 nums[i] == nums[i + 1] ，我们就把 nums[i] 删掉，所以将 result 加一，索引 i 加一继续向前推进遍历 nums ；如果 nums[i] != nums[i + 1]  ，那么我们就将索引 i 加 2 ，继续向前推进遍历 nums 。在遍历结束之后可能 i 还小于 N ，那么说明索引 i 后面的元素都要被去掉，那么将 N-i 加到 result 即可，最后返回 result 即可。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。
### 解答
				
	class Solution(object):
	    def minDeletion(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        if not nums:return 0
	        N = len(nums)
	        i = 0
	        result = 0
	        while i < N-1:
	            if nums[i] == nums[i+1]:
	                result += 1
	                i += 1
	            else:
	                i += 2
	        if i < N:
	            result += (N-i)
	        return result

            	      
			
### 运行结果

	114 / 114 test cases passed.
	Status: Accepted
	Runtime: 1696 ms
	Memory Usage: 25.5 MB

### 原题链接



https://leetcode.com/contest/weekly-contest-286/problems/minimum-deletions-to-make-array-beautiful/


您的支持是我最大的动力
