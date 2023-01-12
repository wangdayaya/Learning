leetcode  1537. Get the Maximum Score（python）

### 描述

You are given two sorted arrays of distinct integers nums1 and nums2.

A valid path is defined as follows:

* Choose array nums1 or nums2 to traverse (from index-0).
* Traverse the current array from left to right.
* If you are reading any value that is present in nums1 and nums2 you are allowed to change your path to the other array. (Only one repeated value is considered in the valid path).

The score is defined as the sum of uniques values in a valid path.Return the maximum score you can obtain of all possible valid paths. Since the answer may be too large, return it modulo 10^9 + 7.



Example 1:

![](https://assets.leetcode.com/uploads/2020/07/16/sample_1_1893.png)
	
	Input: nums1 = [2,4,5,8,10], nums2 = [4,6,8,9]
	Output: 30
	Explanation: Valid paths:
	[2,4,5,8,10], [2,4,5,8,9], [2,4,6,8,9], [2,4,6,8,10],  (starting from nums1)
	[4,6,8,9], [4,5,8,10], [4,5,8,9], [4,6,8,10]    (starting from nums2)
	The maximum is obtained with the path in green [2,4,6,8,10].


	
Example 2:


	Input: nums1 = [1,3,5,7,9], nums2 = [3,5,100]
	Output: 109
	Explanation: Maximum sum is obtained with the path [1,3,5,100].

Example 3:

	Input: nums1 = [1,2,3,4,5], nums2 = [6,7,8,9,10]
	Output: 40
	Explanation: There are no common elements between nums1 and nums2.
	Maximum sum is obtained with the path [6,7,8,9,10].

	
Example 4:

	Input: nums1 = [1,4,5,8,9,11,19], nums2 = [2,3,4,11,12]
	Output: 61




Note:

	1 <= nums1.length, nums2.length <= 10^5
	1 <= nums1[i], nums2[i] <= 10^7
	nums1 and nums2 are strictly increasing.


### 解析


根据题意，给定两个不同整数 nums1 和 nums2 的已经排序的数组。有效路径定义如下：

* 选择要遍历的数组 nums1 或 nums2（从索引 0 开始）。
* 从左到右遍历当前数组。
* 如果您正在读取 nums1 和 nums2 中都存在的某个值，则可以将路径更改为另一个数组。 （在有效路径中只考虑一个重复值）。

分数定义为有效路径中不同值的总和，返回您可以获得的所有可能有效路径的最大分数。 由于答案可能太大，将其取模 10^9 + 7 返回。

其实题意结合例子一还是很好理解的，两个数组的转换位置正好是两个数组都有的数字，这样可以保证有效路径总是升序排列的。最粗暴的方法就是找出所有的可能的路径然后找出拥有最大和的数组。但是只要有两个数组有相同的数字就会分出两条路，这种时间复杂度为 O(2^n) ，肯定是超时的。

其实我们可以用贪心的思想，每当找到两个数组共同拥有的数字的时候，就计算两个数组各自在该数字之前的数字总和，比较之后得到一个较大的和，重复这个过程，如下所示有两个字符串 A 和 B ，其中的 O 表示相同的字符：

* A ： XXXXOZZZZOXXX
* B ：   TTTOSSSSOPPP

现比较 XXXXO 和 TTTO 的各自总和，取最大的值作为找到第一个 O 时候的最大和 a ，然后比较 ZZZZO 和 SSSSO 的各自总和，取最大的值再加 a 作为找到第二个 O 时候的最大和 b ，然后再把各自剩余的数字 XXX 和 PPP 加起来得到 s1 和 s2 ，取较大值即为结果。


### 解答
				
	
	class Solution(object):
	    def maxSum(self, nums1, nums2):
	        """
	        :type nums1: List[int]
	        :type nums2: List[int]
	        :rtype: int
	        """
	        i = j = s1 = s2 = 0
	        m = len(nums1)
	        n = len(nums2)
	        while i<m or j<n:
	            if i==m:
	                s2 += nums2[j]
	                j += 1
	            elif j==n:
	                s1 += nums1[i]
	                i += 1
	            elif nums1[i] < nums2[j]:
	                s1 += nums1[i]
	                i += 1
	            elif nums1[i] > nums2[j]:
	                s2 += nums2[j]
	                j += 1
	            elif nums1[i] == nums2[j]:
	                s1 = max(s1+nums1[i], s2+nums2[j])
	                s2 = s1
	                i += 1
	                j += 1
	        return max(s1, s2)%(10**9+7)
	        
            	      
			
### 运行结果

	Runtime: 484 ms, faster than 88.46% of Python online submissions for Get the Maximum Score.
	Memory Usage: 21.9 MB, less than 42.31% of Python online submissions for Get the Maximum Score.


原题链接：https://leetcode.com/problems/get-the-maximum-score/



您的支持是我最大的动力
