leetcode  986. Interval List Intersections（python）

### 描述

You are given two lists of closed intervals, firstList and secondList, where firstList[i] = [start<sub>i</sub>, end<sub>i</sub>] and secondList[j] = [start<sub>j</sub>, end<sub>j</sub>]. Each list of intervals is pairwise disjoint and in sorted order.

Return the intersection of these two interval lists.

A closed interval \[a, b] (with a <= b) denotes the set of real numbers x with a <= x <= b.

The intersection of two closed intervals is a set of real numbers that are either empty or represented as a closed interval. For example, the intersection of [1, 3] and [2, 4] is [2, 3].



Example 1:

![](https://assets.leetcode.com/uploads/2019/01/30/interval1.png)

	Input: firstList = [[0,2],[5,10],[13,23],[24,25]], secondList = [[1,5],[8,12],[15,24],[25,26]]
	Output: [[1,2],[5,5],[8,10],[15,23],[24,24],[25,25]]

	
Example 2:

	Input: firstList = [[1,3],[5,9]], secondList = []
	Output: []


Example 3:


	Input: firstList = [], secondList = [[4,8],[10,12]]
	Output: []
	
Example 4:


	Input: firstList = [[1,7]], secondList = [[3,10]]
	Output: [[3,7]]
	

Note:

* 	0 <= firstList.length, secondList.length <= 1000
* 	firstList.length + secondList.length >= 1
* 	0 <= start<sub>i</sub> < end<sub>i</sub> <= 109
* 	end<sub>i</sub> < start<sub>i+1</sub>
* 	0 <= start<sub>j</sub> < end<sub>j</sub> <= 109
* 	end<sub>j</sub> < start<sub>j+1</sub>



### 解析


根据题意，给定两个闭区间列表，firstList 和 secondList，其中 firstList[i] = [start<sub>i</sub>, end<sub>i</sub>] 和 secondList[j] = [start<sub>j</sub>, end<sub>j</sub>]。 每个区间列表都是成对不相交的，并按排序顺序排列。题目要求我们返回这两个区间列表的交集。

题目中还给出了闭区间和交集的概念，闭区间 [a, b]（a <= b）表示实数 x 的集合，a <= x <= b。两个闭区间的交集是一组实数，它们要么为空，要么表示为闭区间。 例如，[1, 3] 和 [2, 4] 的交集是 [2, 3]。

思路很简单，直接遍历所有的闭区间找交集即可：

* 如果 A 或者 B 有一个为空列表，直接返回空列表
* 初始化两个指针 i 和 j 都为 0 ，结果 result 为空列表
* 当 i 小于 A 的长度并且 j 小于 B 的长度时候一直进行 while 循环

		如果 A[i][1] < B[j][0] 说明 A[i] 和 B[j] 没有交集，i 加一进行下一次循环
		同理 B[j][1] < A[i][0] 也说明  A[i] 和 B[j] 没有交集 ，j 加一进行下一次循环
		如果有交集，直接将其存入 result 中
		如果 A[i][1] < B[j][1] 说明需要查看 A[i+1] 与 B[j] 是否存在交集，所以  i 加一，否则 j 加一，一样的道理
* 循环结束返回 result 即可。


### 解答
				

	class Solution(object):
	    def intervalIntersection(self, A, B):
	        """
	        :type A: List[List[int]]
	        :type B: List[List[int]]
	        :rtype: List[List[int]]
	        """
	        if not A or not B: return []
	        i = j = 0
	        result = []
	        while i < len(A) and j < len(B):
	            if A[i][1] < B[j][0]:
	                i += 1
	                continue
	            if B[j][1] < A[i][0]:
	                j += 1
	                continue
	            result.append([max(A[i][0], B[j][0]), min(A[i][1], B[j][1])])
	            if A[i][1] < B[j][1]:
	                i += 1
	            else:
	                j += 1
	        return result
            	      
			
### 运行结果

	Runtime: 116 ms, faster than 90.07% of Python online submissions for Interval List Intersections.
	Memory Usage: 14.3 MB, less than 69.98% of Python online submissions for Interval List Intersections.	

### 解析


还可以有简化一下代码，原理同上。

### 解答


	class Solution(object):
	    def intervalIntersection(self, A, B):
	        """
	        :type A: List[List[int]]
	        :type B: List[List[int]]
	        :rtype: List[List[int]]
	        """
	        if not A or not B: return []
	        i = j = 0
	        result = []
	        while i < len(A) and j < len(B):
	            L = max(A[i][0], B[j][0])
	            R = min(A[i][1], B[j][1])
	            if L<=R:
	                result.append([L, R])
	            if A[i][1] < B[j][1]:
	                i += 1
	            else:
	                j += 1
	        return result

### 运行结果
	
	Runtime: 112 ms, faster than 96.69% of Python online submissions for Interval List Intersections.
	Memory Usage: 14.3 MB, less than 69.98% of Python online submissions for Interval List Intersections.



原题链接：https://leetcode.com/problems/interval-list-intersections/



您的支持是我最大的动力
