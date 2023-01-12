leetcode  2333. Minimum Sum of Squared Difference（python）




### 描述

You are given two positive 0-indexed integer arrays nums1 and nums2, both of length n. The sum of squared difference of arrays nums1 and nums2 is defined as the sum of (nums1[i] - nums2[i])2 for each 0 <= i < n. You are also given two positive integers k1 and k2. You can modify any of the elements of nums1 by +1 or -1 at most k1 times. Similarly, you can modify any of the elements of nums2 by +1 or -1 at most k2 times.

Return the minimum sum of squared difference after modifying array nums1 at most k1 times and modifying array nums2 at most k2 times. Note: You are allowed to modify the array elements to become negative integers.

 



Example 1:

	Input: nums1 = [1,2,3,4], nums2 = [2,10,20,19], k1 = 0, k2 = 0
	Output: 579
	Explanation: The elements in nums1 and nums2 cannot be modified because k1 = 0 and k2 = 0. 
	The sum of square difference will be: (1 - 2)^2 + (2 - 10)^2 + (3 - 20)^2 + (4 - 19)^2 = 579.


	
Example 2:

	Input: nums1 = [1,4,10,12], nums2 = [5,8,6,9], k1 = 1, k2 = 1
	Output: 43
	Explanation: One way to obtain the minimum sum of square difference is: 
	- Increase nums1[0] once.
	- Increase nums2[2] once.
	The minimum of the sum of square difference will be: 
	(2 - 5)^2 + (4 - 8)^2 + (10 - 7)^2 + (12 - 9)^2 = 43.
	Note that, there are other ways to obtain the minimum of the sum of square difference, but there is no way to obtain a sum smaller than 43.




Note:

	n == nums1.length == nums2.length
	1 <= n <= 10^5
	0 <= nums1[i], nums2[i] <= 10^5
	0 <= k1, k2 <= 10^9


### 解析

根据题意，给定两个长度为 n 的正 0 索引整数数组 nums1 和 nums2 。数组 nums1 和 nums2 的平方差之和定义为每个 0 <= i < n 的 (nums1[i] - nums2[i])^2 之和。还给定两个正整数 k1 和 k2 。 要求我们可以最多对 nums1 进行 k1 次操作，最多对 nums2 进行 k2 次操作，每次操作可以对任何元素 +1 或 -1 。返回数组 nums1 最多修改 k1 次和数组 nums2 最多修改 k2 次后的最小平方和。

其实我们经过举例可以发现，要想使得最后的结果最小，我们只需要将每个位置对应的元素 abs(nums1[i] - nums2[i]) 的值越小越好，我们虽然规定只对 nums1 最多进行 k1 次操作，对 nums2 进行最多 k2 次操作，但是其实我们只需要知道 k1 + k2 即可，这有两种情况：

* 如果 k1+k2 的值大于等于所有绝对值的和，我们可以将所有的绝对值都变为 0 ，这样就可以直接返回结果 0 。
* 如果 k1+k2 的值小于所有绝对值的和，我们利用有序的数据结构，每次尽可能多的将最大的绝对值进行减一操作即可，然后更新有序的数据结构。不断进行操作知道 k1+k2 变为 0 ，我们直接计算最后的结果即可。

时间复杂度为 O(NlogN) ，空间复杂度为 O(N) 。

### 解答


	class Solution(object):
	    def minSumSquareDiff(self, nums1, nums2, k1, k2):
	        """
	        :type nums1: List[int]
	        :type nums2: List[int]
	        :type k1: int
	        :type k2: int
	        :rtype: int
	        """
	        from sortedcontainers import SortedDict
	        k = k1 + k2
	        absL = []
	        for i in range(len(nums1)):
	            absL.append(abs(nums1[i] - nums2[i]))
	        if k >= sum(absL):
	            return 0
	        S = SortedDict(collections.Counter(absL))
	        while k > 0:
	            key, value = S.popitem(-1)
	            if k >= value:
	                if key - 1 in S:
	                    S[key - 1] += value
	                else:
	                    S[key - 1] = value
	                k -= value
	            else:
	                S[key] = value - k
	                if key - 1 in S:
	                    S[key - 1] += k
	                else:
	                    S[key - 1] = k
	                k = 0
	        result = 0
	        for key, value in S.items():
	            result += key * key * value
	        return result
	        
### 运行结果

	37 / 37 test cases passed.
	Status: Accepted
	Runtime: 2080 ms
	Memory Usage: 34 MB

### 原题链接

	https://leetcode.com/contest/biweekly-contest-82/problems/minimum-sum-of-squared-difference/


您的支持是我最大的动力
