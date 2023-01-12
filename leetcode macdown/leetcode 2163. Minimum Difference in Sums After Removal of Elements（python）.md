

### 前言

这是 Biweekly Contest 71 比赛的第四题，难度 Hard  ，考察的是对堆数据结构（或者优先队列）的使用。

### 描述

You are given a 0-indexed integer array nums consisting of 3 * n elements.

You are allowed to remove any subsequence of elements of size exactly n from nums. The remaining 2 * n elements will be divided into two equal parts:

* The first n elements belonging to the first part and their sum is sum<sub>first</sub>.
* The next n elements belonging to the second part and their sum is sum<sub>second</sub>.

The difference in sums of the two parts is denoted as sum<sub>first</sub> - sum<sub>second</sub>.

* For example, if sum<sub>first</sub> = 3 and sum<sub>second</sub> = 2, their difference is 1.
* Similarly, if sum<sub>first</sub> = 2 and sum<sub>second</sub> = 3, their difference is -1.

Return the minimum difference possible between the sums of the two parts after the removal of n elements.



Example 1:


	Input: nums = [3,1,2]
	Output: -1
	Explanation: Here, nums has 3 elements, so n = 1. 
	Thus we have to remove 1 element from nums and divide the array into two equal parts.
	- If we remove nums[0] = 3, the array will be [1,2]. The difference in sums of the two parts will be 1 - 2 = -1.
	- If we remove nums[1] = 1, the array will be [3,2]. The difference in sums of the two parts will be 3 - 2 = 1.
	- If we remove nums[2] = 2, the array will be [3,1]. The difference in sums of the two parts will be 3 - 1 = 2.
	The minimum difference between sums of the two parts is min(-1,1,2) = -1. 
	



Note:


	nums.length == 3 * n
	1 <= n <= 10^5
	1 <= nums[i] <= 10^5

### 解析

根据题意，给定一个由 3 * n 个元素组成的 0 索引整数数组 nums。可以从 nums 中删除大小正好为 n 的元素的任何子序列。 剩下的 2 * n 个元素将被分成两等份：

* 属于第一部分的前 n 个元素，它们的和是 sum<sub>first</sub>。
* 接下来的 n 个元素属于第二部分，它们的和是 sum<sub>second</sub>。

两部分之和的差值表示为 sum<sub>first</sub> - sum<sub>second</sub>。例如，如果 sum<sub>first</sub> = 3 和 sum<sub>second</sub> = 2，则它们的差为 1。返回删除 n 个元素后两部分之和之间可能的最小差异。

我一开始没什么思路，看了[大佬的解答过程](https://leetcode.com/problems/minimum-difference-in-sums-after-removal-of-elements/discuss/1747029/Python-Explanation-with-pictures-Priority-Queue.)，真的是简单易懂，内容简洁。我这里就再把大佬的思路理清一下。

这道题其实可以转化为另外一个问题，那就是要将 nums 分为前后两个部分，要想让最后的差值最小，那就让前面的部分的和最小，后面部分的和最大。

开始我们先找到 N = len(nums)//3 ，这里用到了堆数据结构，我们先在 nums[N-1:2\*N] 范围内从左到右每个位置 i ，找出 nums[:i] 范围内的 N 个最小的数字之和，将其加入到 preMin 。同理我们用堆数据结构，再在 nums[N:2\*N+1] 范围内从右到左每个位置 i ，找出 nums[i:] 范围内的 N 个最大的数字之和，将其加入到  sufMax ，在得到 sufMax 之后需要将 sufMax 进行反转。然后同时遍历 preMin 和 sufMax 相同位置的元素，计算 preMin-sufMax ，其实也就是在第 i 个位置时，nums[:i] 可以得到的最小值减 nums[i:] 可以得到的最大值，然后找出差值的最小值返回即可。

时间复杂度为 O(NlogN) ，刚刚好没有超时，空间复杂度为 O(N) 。
### 解答
				
	class Solution(object):
	    def minimumDifference(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        N = len(nums)//3
	        # preMin
	        preMin = [sum(nums[:N])]
	        curMin = sum(nums[:N])
	        pre_hp = [-x for x in nums[:N]]
	        heapq.heapify(pre_hp)
	        for i in range(N, 2*N):
	            val = -heapq.heappop(pre_hp)
	            curMin -= val
	            tmp = min(val, nums[i])
	            curMin += tmp
	            preMin.append(curMin)
	            heapq.heappush(pre_hp, -tmp)
	        # sufMax
	        sufMax = [sum(nums[-N:])]
	        curMax = sum(nums[-N:])
	        suf_hp = [x for x in nums[2*N:]]
	        heapq.heapify(suf_hp)
	        for i in range(2*N-1, N-1, -1):
	            val = heapq.heappop(suf_hp)
	            curMax -= val
	            tmp = max(val, nums[i])
	            curMax += tmp
	            sufMax.append(curMax)
	            heapq.heappush(suf_hp, tmp)
	        # find min diff    
	        sufMax = sufMax[::-1]
	        result = float('inf')
	        for a,b in zip(preMin, sufMax):
	            result = min(result, a-b)
	        return result
	        
	        
            	      
			
### 运行结果


	109 / 109 test cases passed.
	Status: Accepted
	Runtime: 4865 ms
	Memory Usage: 44.2 MB

### 原题链接



https://leetcode.com/contest/biweekly-contest-71/problems/minimum-difference-in-sums-after-removal-of-elements/


您的支持是我最大的动力
