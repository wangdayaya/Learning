leetcode  462. Minimum Moves to Equal Array Elements II（python）




### 描述


Given an integer array nums of size n, return the minimum number of moves required to make all array elements equal.

In one move, you can increment or decrement an element of the array by 1.

Test cases are designed so that the answer will fit in a 32-bit integer.


Example 1:


	Input: nums = [1,2,3]
	Output: 2
	Explanation:
	Only two moves are needed (remember each move increments or decrements one element):
	[1,2,3]  =>  [2,2,3]  =>  [2,2,2]
	
Example 2:

	Input: nums = [1,10,2,9]
	Output: 16



Note:

	n == nums.length
	1 <= nums.length <= 10^5
	-10^9 <= nums[i] <= 10^9


### 解析

根据题意，给定一个大小为 n 的整数数组 nums ，返回使所有数组元素相等所需的最小操作次数。每一次操作中，您可以将数组的元素加一或减一 。

根据我们最朴素的想法，在 nums 最大值和最小值之间的某个数字，这个数字在不停地增大和缩小，我们可以算出移动步数也在增大或者缩小，这时我们就能找到规律，那就是以 nums 中中位数为基准，然后对其他数字进行增大和缩小的总次数是最少的。

证明，假如我们此时有一个 nums ，k 表示我们要定的基准，比 k 小的数字有 m 个，比 k 大的个数有 n 个，假如此时我们的总的操作次数是 steps ，那么我们将 k 往上加一 ，此时我们的总的操作次数就是 steps + m - n ，很明显当 n 大于等于 m 的时候，能让 steps 变小，所以我们从最下面不断往上移动 k ，steps 可以一直减小，直到我们将 k 移动到 n 等于 m 的时候，我们 steps 此时最小，而这时候的 k 就是中位数。

![](https://pic.leetcode-cn.com/1652894015-mkUqDd-%E5%B9%BB%E7%81%AF%E7%89%871.PNG)

时间复杂度为 O(NlogN) ，空间复杂度为 O(1)。



### 解答
				

	class Solution:
	    def minMoves2(self, nums: List[int]) -> int:
	        result = 0
	        nums.sort()
	        mid = nums[len(nums)//2]
	        for n in nums:
	            result += abs(mid-n)
	        return result
            	      
			
### 运行结果

	Runtime: 162 ms, faster than 9.21% of Python3 online submissions for Minimum Moves to Equal Array Elements II.
	Memory Usage: 15.5 MB, less than 33.43% of Python3 online submissions for Minimum Moves to Equal Array Elements II.



### 原题链接

https://leetcode.com/problems/minimum-moves-to-equal-array-elements-ii/

您的支持是我最大的动力
