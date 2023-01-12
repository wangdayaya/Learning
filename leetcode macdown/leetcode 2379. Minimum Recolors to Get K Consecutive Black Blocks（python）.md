leetcode  2379. Minimum Recolors to Get K Consecutive Black Blocks（python）




### 描述

You are given a 0-indexed string blocks of length n, where blocks[i] is either 'W' or 'B', representing the color of the ith block. The characters 'W' and 'B' denote the colors white and black, respectively. You are also given an integer k, which is the desired number of consecutive black blocks. In one operation, you can recolor a white block such that it becomes a black block.

Return the minimum number of operations needed such that there is at least one occurrence of k consecutive black blocks.



Example 1:

	Input: blocks = "WBBWWBBWBW", k = 7
	Output: 3
	Explanation:
	One way to achieve 7 consecutive black blocks is to recolor the 0th, 3rd, and 4th blocks
	so that blocks = "BBBBBBBWBW". 
	It can be shown that there is no way to achieve 7 consecutive black blocks in less than 3 operations.
	Therefore, we return 3.

	
Example 2:

	Input: blocks = "WBWBBBW", k = 2
	Output: 0
	Explanation:
	No changes need to be made, since 2 consecutive black blocks already exist.
	Therefore, we return 0.


Note:


	n == blocks.length
	1 <= n <= 100
	blocks[i] is either 'W' or 'B'.
	1 <= k <= n

### 解析

根据题意，给定一个长度为 n 的 0 索引字符串 blocks ，其中 blocks[i] 是“W”或“B”，代表第 i 个块的颜色。 字符“W”和“B”分别表示白色和黑色。给定一个整数 k，它是所需的连续黑色块数。在一次操作中，我们可以重新着色白色块，使其变为黑色块。返回所需的最小操作数，以便至少出现 k 个连续的黑色块。

这道题直接使用滑动窗口的思想解题即可，因为已经告诉我们必须有 k 个连续的黑色块出现，所以我们每次取 k 大小的窗口在 blocks 上从左到右滑动，并且使用 result 来记录当前如果变成全黑色进行的最小操作并不断更新，遍历结束我们直接返回 result 即可。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def minimumRecolors(self, blocks, k):
	        """
	        :type blocks: str
	        :type k: int
	        :rtype: int
	        """
	        result = k
	        for i in range(0, len(blocks)-k+1):
	            result = min(result, blocks[i:i+k].count('W'))
	        return result

### 运行结果

	122 / 122 test cases passed.
	Status: Accepted
	Runtime: 19 ms
	Memory Usage: 13.4 MB


### 原题链接

https://leetcode.com/contest/biweekly-contest-85/problems/minimum-recolors-to-get-k-consecutive-black-blocks/


您的支持是我最大的动力
