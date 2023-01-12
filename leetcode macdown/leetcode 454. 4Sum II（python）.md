leetcode  454. 4Sum II（python）




### 描述


Given four integer arrays nums1, nums2, nums3, and nums4 all of length n, return the number of tuples (i, j, k, l) such that:

* 0 <= i, j, k, l < n
* nums1[i] + nums2[j] + nums3[k] + nums4[l] == 0



Example 1:


	Input: nums1 = [1,2], nums2 = [-2,-1], nums3 = [-1,2], nums4 = [0,2]
	Output: 2
	Explanation:
	The two tuples are:
	1. (0, 0, 0, 1) -> nums1[0] + nums2[0] + nums3[0] + nums4[1] = 1 + (-2) + (-1) + 2 = 0
	2. (1, 1, 0, 0) -> nums1[1] + nums2[1] + nums3[0] + nums4[0] = 2 + (-1) + (-1) + 0 = 0
	
Example 2:


	Input: nums1 = [0], nums2 = [0], nums3 = [0], nums4 = [0]
	Output: 1




Note:

	n == nums1.length
	n == nums2.length
	n == nums3.length
	n == nums4.length
	1 <= n <= 200
	-2^28 <= nums1[i], nums2[i], nums3[i], nums4[i] <= 2^28


### 解析

根据题意，给定四个长度为 n 的整数数组 A、B、C 和 D，返回元组 (i, j, k, l) 的数量，使得：

* 0 <= i, j, k, l < n
* A[i] + B[j] + C[k] + D[l] == 0

其实题意很容易理解，正常来说四次 for 循环就能解题，但是这种 O(N^4) 的时间复杂度基本上没什么戏，我们看限制条件每个数组的长度最长也就是 200 ，如果是四重循环，最大可能的时间复杂度为 O(10^8)，按照 leetcode 常见的运行条件，所以肯定是会超时的，空间复杂度为 O(1) 。

我们可以使用字典将使用两重 for 循环，将 A 、B 两个数组中的元素加起来的值存入字典 t 中，然后再使用两重 for 循环，将 C 、D 两个数组中的元素加起来的值的负值，去 t 中查找，如果存在则结果 result 加一，遍历结束得到的 result 即为结果，时间复杂度也能降到 O(N^2) ，空间复杂度为 O(n) 。

### 解答

	class Solution(object):
	    def fourSumCount(self, A, B, C, D):
	        """
	        :type A: List[int]
	        :type B: List[int]
	        :type C: List[int]
	        :type D: List[int]
	        :rtype: int
	        """
	        result = 0
	        for a in A:
	            for b in B:
	                for c in C:
	                    for d in D:
	                        if a+b+c+d == 0:
	                            result += 1
	        return result
### 运行结果

	Time Limit Exceeded
 
### 解答
				

	class Solution(object):
	    def fourSumCount(self, A, B, C, D):
	        """
	        :type A: List[int]
	        :type B: List[int]
	        :type C: List[int]
	        :type D: List[int]
	        :rtype: int
	        """
	        t = {}
	        for a in A:
	            for b in B:
	                t[a+b] = t.get(a+b,0)+1
	        result = 0
	        for c in C:
	            for d in D:
	                if -(c+d) in t:
	                    result+=t[-(c+d)]
	        return result
            	      
			
### 运行结果


	Runtime: 1188 ms, faster than 17.74% of Python online submissions for 4Sum II.
	Memory Usage: 13.6 MB, less than 94.52% of Python online submissions for 4Sum II.

### 原题链接


https://leetcode.com/problems/4sum-ii/


您的支持是我最大的动力
