leetcode 2239. Find Closest Number to Zero （python）


这道题是第 76 场 leetcode 双周赛的第一题，难度为 Eazy ，主要考察的是对列表的排序

### 描述


Given an integer array nums of size n, return the number with the value closest to 0 in nums. If there are multiple answers, return the number with the largest value.


Example 1:

	Input: nums = [-4,-2,1,4,8]
	Output: 1
	Explanation:
	The distance from -4 to 0 is |-4| = 4.
	The distance from -2 to 0 is |-2| = 2.
	The distance from 1 to 0 is |1| = 1.
	The distance from 4 to 0 is |4| = 4.
	The distance from 8 to 0 is |8| = 8.
	Thus, the closest number to 0 in the array is 1.

	
Example 2:


	Input: nums = [2,-1,1]
	Output: 1
	Explanation: 1 and -1 are both the closest numbers to 0, so 1 being larger is returned.




Note:

	1 <= n <= 1000
	-10^5 <= nums[i] <= 10^5


### 解析


根据题意，给定一个大小为 n 的整数数组 nums ，返回 nums 中值最接近 0 的数字。 如果有多个答案，则返回具有最大值的数字。

这道题的题目很简单，考察的是对包含了二元组的列表的排序，我在比赛的时候是直接根据题意，先找出所有元素与 0 的距离，然后再找出其中最小的距离值 mn ，最后再对 nums 中距离值都为 mn 的元素进行排序，取最大值返回即可。这种解法虽然没有问题可以 AC ，但是代码是又臭又长，耗时还比较多，比赛的时候就是为了节省时间，不想花费过多的时间去思考。

现在我们对解法进行优化，其实就是遍历 nums 中的所有元素，将每个元素转换为一个二元组，二元组中第一个元素是与 0 的距离，第二个元素是元素本身值，我们要让第一个元素最小，第二个元素最大，这样我们最后就能得到答案。这种解法代码简洁易懂，很巧妙。

时间复杂度为 O(N) ，空间复杂为 O(N) 。


### 解答
				

	class Solution(object):
	    def findClosestNumber(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        result = [[-abs(n), n] for n in nums]
	        return max(result)[1]
	        
            	      
			
### 运行结果

	224 / 224 test cases passed.
	Status: Accepted
	Runtime: 116 ms
	Memory Usage: 13.7 MB


### 原题链接



https://leetcode.com/contest/biweekly-contest-76/problems/find-closest-number-to-zero/

您的支持是我最大的动力
