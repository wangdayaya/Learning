leetcode  2145. Count the Hidden Sequences（python）

### 前言

参加了第 70 场双周赛，题目还是有点难度的，榜单第一名四道题做完都用了 17 分钟，之前的单周赛第一名只是 7、8 分钟就结束了。这是 Biweekly Contest 70 的第二题，难度 Medium ，其实就是找规律题，感觉没什么考察的点。


### 描述

You are given a 0-indexed array of n integers differences, which describes the differences between each pair of consecutive integers of a hidden sequence of length (n + 1). More formally, call the hidden sequence hidden, then we have that differences[i] = hidden[i + 1] - hidden[i].

You are further given two integers lower and upper that describe the inclusive range of values [lower, upper] that the hidden sequence can contain.

* For example, given differences = [1, -3, 4], lower = 1, upper = 6, the hidden sequence is a sequence of length 4 whose elements are in between 1 and 6 (inclusive).
* [3, 4, 1, 5] and [4, 5, 2, 6] are possible hidden sequences.
* [5, 6, 3, 7] is not possible since it contains an element greater than 6.
* [1, 2, 3, 4] is not possible since the differences are not correct.

Return the number of possible hidden sequences there are. If there are no possible sequences, return 0.



Example 1:

	Input: differences = [1,-3,4], lower = 1, upper = 6
	Output: 2
	Explanation: The possible hidden sequences are:
	- [3, 4, 1, 5]
	- [4, 5, 2, 6]
	Thus, we return 2.

	
Example 2:

	Input: differences = [3,-4,5,1,-2], lower = -4, upper = 5
	Output: 4
	Explanation: The possible hidden sequences are:
	- [-3, 0, -4, 1, 2, 0]
	- [-2, 1, -3, 2, 3, 1]
	- [-1, 2, -2, 3, 4, 2]
	- [0, 3, -1, 4, 5, 3]
	Thus, we return 4.



Note:

	n == differences.length
	1 <= n <= 10^5
	-10^5 <= differences[i] <= 10^5
	-10^5 <= lower <= upper <= 10^5


### 解析


根据题意，给定一个包含 n 个整数的 0 索引数组 differences ，它描述了长度为 n + 1 的某个隐藏序列 hidden 的每对连续整数之间的差异，差异规律就是 differences[i] = hidden[i + 1] - hidden[i]。同时又给出两个整数 lower 和 upper，它们描述了隐藏序列 hidden 可以包含的值在 [lower, upper] 范围内。返回可能存在的隐藏序列的数量。 如果没有可能的序列，则返回 0。

* 例如，给定 differences = [1, -3, 4] ，lower = 1，upper = 6 ，隐藏序列是长度为 4 的序列，其元素介于 1 和 6 之间。
* [3, 4, 1, 5] 和 [4, 5, 2, 6] 是可能的隐藏序列。
* [5, 6, 3, 7] 是不可能的，因为它包含大于 6 的元素。
* [1, 2, 3, 4] 是不可能的，因为差异不正确。

乍一看这个题没有思路，最暴力的方法肯定行不通，因为题目限制条件 differences.length 的最大值为 10^5 ，而暴力解法需要两层循环，时间复杂度为 O(n^2) ，肯定会超时，这个条件就是限制我们肯定是要使用 O(n) 以内甚至 O(1) 的解法，肯定是需要我们找规律来解题的：

* 首先我们发现 differences 中的差值绝对值肯定要小于等于 upper - lower ，否则肯定无法有相应的  hidden 出现
* 我们发现每个 differences 是后一个元素减前一个元素的差值，我们遍历并计算一次，进一步可以得到后一个元素与第一个元素的差值是多少，这样所有的元素都与第一个元素的大小有关系，所以我们就将题目转化为第一个元素的取值范围，只要第一个元素的取值在合理的范围之内，那么整个的 hidden 肯定是合理的，这里的合理指的是在 hidden 元素都在 [lower, upper] 范围内，且不重复。
* 在新得到的 differences 中，我们可以找到最大值和最小值 d\_mn 和 d\_mx ，分别表示了 hidden 中与第一个元素相差最多和最少的距离。这样我们就知道第一个元素的可选最小值为 max(lower, lower-d_mn) ，可选的最大值为 min(upper, upper-d_mx) ，直接计数就能得到第一个元素的可选值个数，也就是 hidden 的可能序列个数。

其实这道题冷静思考做出来还是很容易的，但是比赛的时候我去想着优化暴力解法，结果浪费了不少时间，直到看到限制条件才果断放弃了暴力解法，去找规律的，所以还是要冷静看题，着急写代码没有卵用。


### 解答
				
	class Solution(object):
	    def numberOfArrays(self, differences, lower, upper):
	        """
	        :type differences: List[int]
	        :type lower: int
	        :type upper: int
	        :rtype: int
	        """
	        if max(abs(min(differences)), abs(max(differences))) > upper - lower:
	            return 0
	        n = len(differences)
	        for i in range(1,n):
	            differences[i] = differences[i-1] + differences[i]
	        result = 0
	        d_mn = min(differences)
	        d_mx = max(differences)
	        N = max(lower, lower-d_mn)
	        M = min(upper, upper-d_mx)
	        for i in range(N, M+1):
	            result += 1
	        return result

            	      
			
### 运行结果

	82 / 82 test cases passed.
	Status: Accepted
	Runtime: 1461 ms
	Memory Usage: 26.3 MB

### 原题链接

https://leetcode.com/contest/biweekly-contest-70/problems/count-the-hidden-sequences/



您的支持是我最大的动力
