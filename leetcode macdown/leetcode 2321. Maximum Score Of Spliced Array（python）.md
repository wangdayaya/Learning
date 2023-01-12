leetcode  2321. Maximum Score Of Spliced Array（python）




### 描述


You are given two 0-indexed integer arrays nums1 and nums2, both of length n. You can choose two integers left and right where 0 <= left <= right < n and swap the subarray nums1[left...right] with the subarray nums2[left...right].

* For example, if nums1 = [1,2,3,4,5] and nums2 = [11,12,13,14,15] and you choose left = 1 and right = 2, nums1 becomes [1,12,13,4,5] and nums2 becomes [11,2,3,14,15].

You may choose to apply the mentioned operation once or not do anything. The score of the arrays is the maximum of sum(nums1) and sum(nums2), where sum(arr) is the sum of all the elements in the array arr. Return the maximum possible score. A subarray is a contiguous sequence of elements within an array. arr[left...right] denotes the subarray that contains the elements of nums between indices left and right (inclusive).


Example 1:

	Input: nums1 = [60,60,60], nums2 = [10,90,10]
	Output: 210
	Explanation: Choosing left = 1 and right = 1, we have nums1 = [60,90,60] and nums2 = [10,60,10].
	The score is max(sum(nums1), sum(nums2)) = max(210, 80) = 210.

	
Example 2:

	Input: nums1 = [20,40,20,70,30], nums2 = [50,20,50,40,20]
	Output: 220
	Explanation: Choosing left = 3, right = 4, we have nums1 = [20,40,20,40,20] and nums2 = [50,20,50,70,30].
	The score is max(sum(nums1), sum(nums2)) = max(140, 220) = 220.


Example 3:


	Input: nums1 = [7,11,13], nums2 = [1,1,1]
	Output: 31
	Explanation: We choose not to swap any subarray.
	The score is max(sum(nums1), sum(nums2)) = max(31, 3) = 31.


Note:

	n == nums1.length == nums2.length
	1 <= n <= 10^5
	1 <= nums1[i], nums2[i] <= 10^4


### 解析

根据题意，给定两个长度为 n 的 0 索引整数数组 nums1 和 nums2。 我们可以选择左右两个整数 left 和 right （0 <= left <= right < n ），并将子数组 nums1[left...right] 与子数组 nums2[left...right] 进行交换。

* 例如，如果 nums1 = [1,2,3,4,5] 和 nums2 = [11,12,13,14,15] 并且您选择 left = 1 和 right = 2，则 nums1 变为 [1,12, 13,4,5] 和 nums2 变为 [11,2,3,14,15]。

我们可以选择进行一次上述操作或不执行任何操作。数组的分数是 sum(nums1) 和 sum(nums2) 之间的最大值，返回可能的最大分数。 

假设 A 和 B 都是有 5 个元素的列表，现在的 A 列表和为 s ， A 在经过 swap(2, 3) 操作之后的，其和 s' 变为：

* 	s' = s - a<sub>2</sub> - a<sub>3</sub> + b<sub>2</sub> + <sub>b3</sub>
* 	s' = s + (b<sub>2</sub> - a<sub>2</sub>) + (b<sub>3</sub> - a<sub>3</sub>)
* 	s' = s + diff<sub>2</sub> + diff<sub>3</sub>

可以看出这里的 2 可以换成 left ，3 换成 right：

* 	s' = s + diff<sub>left</sub> + ... +diff<sub>right</sub>

所以要想使 A 的和 s' 最大，就要尽量使得上面公式后半部分的   diff<sub>left</sub> + ... +diff<sub>right</sub> 的值最大，我们分别计算出 nums1 和 nums2 的最大值，然后选出较大的那个返回即可。
	
 

时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def maximumsSplicedArray(self, nums1, nums2):
	        """
	        :type nums1: List[int]
	        :type nums2: List[int]
	        :rtype: int
	        """
	        def s(L1, L2):
	            tmp = result = 0
	            for x, y in zip(L1, L2):
	                tmp += y - x
	                if tmp < 0:
	                    tmp = 0
	                else:
	                    result = max(result, tmp)
	            return sum(L1) + result
	        return max(s(nums1, nums2), s(nums2, nums1))

### 运行结果

	34 / 34 test cases passed.
	Status: Accepted
	Runtime: 1350 ms
	Memory Usage: 36.2 MB
	Submitted: 0 minutes ago


### 原题链接

https://leetcode.com/contest/weekly-contest-299/problems/maximum-score-of-spliced-array/


您的支持是我最大的动力
