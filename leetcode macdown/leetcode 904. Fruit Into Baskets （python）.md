




### 描述

You are visiting a farm that has a single row of fruit trees arranged from left to right. The trees are represented by an integer array fruits where fruits[i] is the type of fruit the ith tree produces. You want to collect as much fruit as possible. However, the owner has some strict rules that you must follow:

* You only have two baskets, and each basket can only hold a single type of fruit. There is no limit on the amount of fruit each basket can hold.
* Starting from any tree of your choice, you must pick exactly one fruit from every tree (including the start tree) while moving to the right. The picked fruits must fit in one of your baskets.
* Once you reach a tree with fruit that cannot fit in your baskets, you must stop.

Given the integer array fruits, return the maximum number of fruits you can pick.



Example 1:

	Input: fruits = [1,2,1]
	Output: 3
	Explanation: We can pick from all 3 trees.

	
Example 2:

	Input: fruits = [0,1,2,2]
	Output: 3
	Explanation: We can pick from trees [1,2,2].
	If we had started at the first tree, we would only pick from trees [0,1].


Example 3:


	Input: fruits = [1,2,3,2,2]
	Output: 4
	Explanation: We can pick from trees [2,3,2,2].
	If we had started at the first tree, we would only pick from trees [1,2].


Note:

	1 <= fruits.length <= 10^5
	0 <= fruits[i] < fruits.length


### 解析

我们正在参观一个农场，该农场从左到右排列着一排果树。树由整数数组 fruits 表示，其中 fruits[i] 是第 i 棵树产生的水果类型。我们想收集尽可能多的水果。但是有一些严格的规则：

* 我们只有两个篮子，每个篮子只能容纳一种水果。每个篮子可以容纳的水果数量没有限制。
* 从选择的任意一颗树开始采摘，必须从每棵树（包括起始树）中只采摘一个水果，采摘的水果应当符合篮子中的水果类型。每采摘一次，你将会向右移动到下一棵树，并继续采摘。
* 一旦你到达一棵树，树里面有的水果无法装进篮子里，必须停下来。

给定整数数组 fruits ，返回我们可以挑选的最大水果数量。

其实这道题考查的就是滑动窗口，因为在给定 fruits 的时候，我们要找的就是一个 fruits 的某个子数组的最长长度，这个子数组里面只有两种水果（两种数字）。我们使用一个字典来统计滑动窗口范围内的水果统计情况，初始化滑动窗口的左右边界 left 和 right ，从左往右遍历 fruits ，right 相应的往右移动位置，这个过程中将出现的水果种类加入字典中，只要字典中有小于等于两种水果种类的合法情况，我就不断更新结果最大的水果数量值 result ，一旦字典中出现的水果种类大于 2 ，那么我们就要不断往右移动 left ，并且将水果种类 fruits[left] 移出字典，同时更新结果 result 。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def totalFruit(self, fruits):
	        """
	        :type fruits: List[int]
	        :rtype: int
	        """
	        cnt = Counter()
	        left = result = 0
	        for right, x in enumerate(fruits):
	            cnt[x] += 1
	            while len(cnt) > 2:
	                cnt[fruits[left]] -= 1
	                if cnt[fruits[left]] == 0:
	                    cnt.pop(fruits[left])
	                left += 1
	            result = max(result, right - left + 1)
	        return result

### 运行结果

	Runtime 965 ms，Beats 30.91%
	Memory 19.7 MB，Beats 24.73%

### 原题链接

https://leetcode.com/problems/fruit-into-baskets/description/


您的支持是我最大的动力
