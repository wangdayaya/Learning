leetcode  2032. Two Out of Three（python）

### 描述


Given three integer arrays nums1, nums2, and nums3, return a distinct array containing all the values that are present in at least two out of the three arrays. You may return the values in any order.



Example 1:

	Input: nums1 = [1,1,3,2], nums2 = [2,3], nums3 = [3]
	Output: [3,2]
	Explanation: The values that are present in at least two arrays are:
	- 3, in all three arrays.
	- 2, in nums1 and nums2.

	
Example 2:

	Input: nums1 = [3,1], nums2 = [2,3], nums3 = [1,2]
	Output: [2,3,1]
	Explanation: The values that are present in at least two arrays are:
	- 2, in nums2 and nums3.
	- 3, in nums1 and nums2.
	- 1, in nums1 and nums3.



Example 3:

	Input: nums1 = [1,2,2], nums2 = [4,3,3], nums3 = [5]
	Output: []
	Explanation: No value is present in at least two arrays.



Note:

	1 <= nums1.length, nums2.length, nums3.length <= 100
	1 <= nums1[i], nums2[j], nums3[k] <= 100


### 解析


根据题意，就是给出了三个数字列表 nums1 , nums2 , nums3 ，题目要求我们返回一个有不同元素的数组，其中包含全部三个数组中至少在两个数组中共同存在的数字。 结果列表可以按任何顺序返回值。

思路很简单，最笨的方法就是暴力循环，将三个列表  nums1 , nums2 , nums3  的值都放入一个集合 nums 中，然后遍历其中所有的元素，如果该元素至少在某两个列表中出现过，那就将其加入结果列表 result 中，遍历结束返回 result 即可。

### 解答
				

	class Solution(object):
	    def twoOutOfThree(self, nums1, nums2, nums3):
	        """
	        :type nums1: List[int]
	        :type nums2: List[int]
	        :type nums3: List[int]
	        :rtype: List[int]
	        """
	        result = []
	        nums = set(nums1 + nums2 + nums3)
	        for num in nums:
	            if (nums1.count(num)>=1 and nums2.count(num)>=1) or (nums1.count(num)>=1 and nums3.count(num)>=1) or (nums2.count(num)>=1 and nums3.count(num)>=1) :
	                result.append(num)
	        return result
	                
            	      
			
### 运行结果


	Runtime: 76 ms, faster than 41.83% of Python online submissions for Two Out of Three.
	Memory Usage: 13.5 MB, less than 59.45% of Python online submissions for Two Out of Three.

### 解析

还有一种解法是看了论坛的高手使用集合的交集和并集来解题，将三个列表两两求交集，得到的肯定是至少在两个列表中出现过的元素，然后将所有交集结果进行求并集即可得到所有的结果。

### 解答

	class Solution(object):
	    def twoOutOfThree(self, nums1, nums2, nums3):
	        """
	        :type nums1: List[int]
	        :type nums2: List[int]
	        :type nums3: List[int]
	        :rtype: List[int]
	        """
	        return list(set(nums1) & set(nums2) | set(nums2) & set(nums3) | set(nums1) & set(nums3))

### 运行结果
	
	Runtime: 44 ms, faster than 98.94% of Python online submissions for Two Out of Three.
	Memory Usage: 13.7 MB, less than 10.40% of Python online submissions for Two Out of Three.
原题链接：https://leetcode.com/problems/two-out-of-three/



您的支持是我最大的动力
