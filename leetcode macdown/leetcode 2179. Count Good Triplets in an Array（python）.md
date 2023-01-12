leetcode  2179. Count Good Triplets in an Array（python）


「这是我参与2022首次更文挑战的第N天，活动详情查看：[2022首次更文挑战](https://juejin.cn/post/7052884569032392740 "https://juejin.cn/post/7052884569032392740")」

### 前言

这是 leetcode 中 Biweekly Contest 72 的第四题，难度为 Hard ，对我反正是有点难度，自己搞了一个小时，硬是最后也没做出来。



### 描述


You are given two 0-indexed arrays nums1 and nums2 of length n, both of which are permutations of [0, 1, ..., n - 1].

A good triplet is a set of 3 distinct values which are present in increasing order by position both in nums1 and nums2. In other words, if we consider pos1<sub>v</sub> as the index of the value v in nums1 and pos2<sub>v</sub> as the index of the value v in nums2, then a good triplet will be a set (x, y, z) where 0 <= x, y, z <= n - 1, such that pos1<sub>x</sub> < pos1<sub>y</sub> < pos1<sub>z</sub> and pos2<sub>x</sub> < pos2<sub>y</sub> < pos2<sub>z</sub>.

Return the total number of good triplets.


Example 1:

	Input: nums1 = [2,0,1,3], nums2 = [0,1,2,3]
	Output: 1
	Explanation: 
	There are 4 triplets (x,y,z) such that pos1x < pos1y < pos1z. They are (2,0,1), (2,0,3), (2,1,3), and (0,1,3). 
	Out of those triplets, only the triplet (0,1,3) satisfies pos2x < pos2y < pos2z. Hence, there is only 1 good triplet.

	
Example 2:

	Input: nums1 = [4,0,1,3,2], nums2 = [4,1,0,2,3]
	Output: 4
	Explanation: The 4 good triplets are (4,0,3), (4,0,2), (4,1,3), and (4,1,2).






Note:

	n == nums1.length == nums2.length
	3 <= n <= 10^5
	0 <= nums1[i], nums2[i] <= n - 1
	nums1 and nums2 are permutations of [0, 1, ..., n - 1].


### 解析

根据题意，给定两个长度为 n 的索引为 0 的数组 nums1 和 nums2，它们都是 [0, 1, ..., n - 1]  的排列。

一个好的三元组是一组 3 个不同的值，它们在 nums1 和 nums2 中按位置递增。 换句话说，如果我们将 pos1<sub>v</sub> 视为 nums1 中值 v 的索引，并将 pos2<sub>v</sub> 视为 nums2 中值 v 的索引，那么一个好的三元组将是一个集合 (x, y, z)，其中 0 <= x y, z <= n - 1，使得 pos1<sub>x</sub> < pos1<sub>y</sub> < pos1<sub>z</sub> 并且 pos2<sub>x</sub> < pos2<sub>y</sub> < pos2<sub>z</sub> 。如例子二所示，我们可以知道 (4,0,3) 在 nums1 中的先后顺序，和 (4,0,3) 在 nums 2 中的先后顺序是一样的，所以是一个合法的好三元组。要求返回好的三元组的总数。

这个解法是看了论坛的大佬解释的，格外详细，其实就是将问题进行了转化，我们要求好的三元组，那我们可以找那些能形成三元组以某个元素 X 为中间元素的组合数量，那这个问题就等价于求，在 nums1 和 nums2 中分别有多少个元素在 X 的左边，在 nums1 和 nums2 中分别有多少个元素在 X 的右边，找出满足相对位置的三元组个数即可。先从左往右进行查找得到 pre ，然后将 nums1 和 nums2 进行反转之后，也从左往右查找得到 suf （这里需要对 suf 进行反转），最后只需要将这两个列表对应位置上面的数字相乘然后求和即可得到答案。

如我们在例子一中，可以得到列表 pre\_a=[0,0,1,3] 记录的是在索引为 i 的时候将 nums1[i] 作为中间值的时候前面可以放置的数字有几个，注意这里可以放置的数字要和在 nums 2 中有相同的前后关系， suf\_a=[1,2,1,0] 记录的时候在索引为 i 的时候，将 num1[i] 作为中间值的时候，后面可以放置的数字有几个，可以放置的数字要和在 nums2 中有相同的前后关系。

大神解答就是这么简单，轮到自己做的时候可是一地鸡毛。



### 解答
				

	from sortedcontainers import SortedList
	class Solution(object):
	    def goodTriplets(self, nums1, nums2):
	        """
	        :type nums1: List[int]
	        :type nums2: List[int]
	        :rtype: int
	        """
	        pos = [0] * len(nums1)
	        for k,v in enumerate(nums2):
	            pos[v] = k
	
	        pos_in_b, pre = SortedList([pos[nums1[0]]]), [0]
	        for n in nums1[1:]:
	            pos_in_b.add(pos[n])
	            pre.append(pos_in_b.bisect_left(pos[n]))
	
	        nums1 = nums1[::-1]
	        nums2 = nums2[::-1]
	        for k,v in enumerate(nums2):
	            pos[v] = k
	        pos_in_b, suf = SortedList([pos[nums1[0]]]), [0]
	        for n in nums1[1:]:
	            pos_in_b.add(pos[n])
	            suf.append(pos_in_b.bisect_left(pos[n]))
	        suf = suf[::-1]
	
	        result = 0
	        for x,y in zip(pre, suf):
	            result += x*y
	        return result
			
### 运行结果

	148 / 148 test cases passed.
	Status: Accepted
	Runtime: 2604 ms
	Memory Usage: 42.2 MB
	
### 原题链接



https://leetcode.com/contest/biweekly-contest-72/problems/count-good-triplets-in-an-array/

您的支持是我最大的动力
