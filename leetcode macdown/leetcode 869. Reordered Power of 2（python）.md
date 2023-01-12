leetcode 869. Reordered Power of 2 （python）




### 描述


You are given an integer n. We reorder the digits in any order (including the original order) such that the leading digit is not zero.

Return true if and only if we can do this so that the resulting number is a power of two.


Example 1:


	Input: n = 1
	Output: true
	
Example 2:

	Input: n = 10
	Output: false





Note:

	1 <= n <= 10^9


### 解析

根据题意，给定一个整数 n 。 我们以任何顺序（包括原始顺序）对 n 中的所有数字重新排序，以使前导数字不为零。如果这样的序列组成的数字大小是 2 的幂时返回 true ，否则返回 false 。

其实这道题使用打表法就能完成，因为 n 最大是 10^9 ，所以我们可以找出来在这个范围内的所有的正整数的 2 的幂次方的数字，最小为 2^0 ，最大为 2^29 ，也就是只有 30 种情况，然后将这些数字都保存在字典 d 中，并记录了各自的数字组成情况，然后我们只需要计算出 n 的数字组成情况，然后和 d 中记录的各种情况进行比对，如果相等说明可以完成题目要求，直接返回 True ，否则说明无法重新排列组成 2 的幂次方的数字，返回 False 即可。

时间复杂度为 O(30) ，也就是 O(1) ，空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def reorderedPowerOf2(self, n):
	        """
	        :type n: int
	        :rtype: bool
	        """
	        d = { pow(2,i) : collections.Counter(str(pow(2,i))) for i in range(0,30)}
	        tmp = collections.Counter(str(n))
	        for k, cnt in d.items():
	            if tmp == cnt:
	                return True
	        return False

### 运行结果

	Runtime: 41 ms, faster than 57.89% of Python online submissions for Reordered Power of 2.
	Memory Usage: 13.3 MB, less than 68.42% of Python online submissions for Reordered Power of 2.

### 原题链接

	https://leetcode.com/problems/reordered-power-of-2/


您的支持是我最大的动力
