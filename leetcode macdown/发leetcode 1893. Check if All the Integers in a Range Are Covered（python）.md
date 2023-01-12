leetcode  1893. Check if All the Integers in a Range Are Covered（python）

### 描述


You are given a 2D integer array ranges and two integers left and right. Each ranges[i] = [start<sub>i</sub>, end<sub>i</sub>] represents an inclusive interval between start<sub>i</sub> and end<sub>i</sub>.

Return true if each integer in the inclusive range [left, right] is covered by at least one interval in ranges. Return false otherwise.

An integer x is covered by an interval ranges[i] = [start<sub>i</sub>, end<sub>i</sub>] if start<sub>i</sub> <= x <= end<sub>i</sub>.


Example 1:
	
	Input: ranges = [[1,2],[3,4],[5,6]], left = 2, right = 5
	Output: true
	Explanation: Every integer between 2 and 5 is covered:
	- 2 is covered by the first range.
	- 3 and 4 are covered by the second range.
	- 5 is covered by the third range.

	
Example 2:


	Input: ranges = [[1,10],[10,20]], left = 21, right = 21
	Output: false
	Explanation: 21 is not covered by any range.





Note:

1 <= ranges.length <= 50

1 <= start<sub>i</sub> <= end<sub>i</sub> <= 50

1 <= left <= right <= 50


### 解析


根据题意，就是给出一个列表 ranges ，里面都是一个个范围，判断 [left, right] 中的每个数字至少都被一个 range 所包含，如果 [left, right] 中的每个数字都至少被一个范围覆盖直接返回 True ， 否则返回 False 。思路简单，直接看代码 。
### 解答
				

	class Solution(object):
	    def isCovered(self, ranges, left, right):
	        """
	        :type ranges: List[List[int]]
	        :type left: int
	        :type right: int
	        :rtype: bool
	        """
	        for i in range(left, right + 1):
	            count = 0
	            for r in ranges:
	                count += 1
	                if r[0] <= i <= r[1]:
	                    break
	                elif count==len(ranges):
	                    return False
	        return True
	            	      
			
### 运行结果

	Runtime: 28 ms, faster than 75.57% of Python online submissions for Check if All the Integers in a Range Are Covered.
	Memory Usage: 13.2 MB, less than 89.31% of Python online submissions for Check if All the Integers in a Range Are Covered.


### 解析


另外可以将 ranges 表示的范围中的数字都放入一个集合 a 中，将 [left,right] 中的数字放入一个集合 b 中，然后求两个集合中间的交集是否等于 b ，如果等于直接返回 True ，否则返回 False 。

### 解答
				
	class Solution(object):
	    def isCovered(self, ranges, left, right):
	        """
	        :type ranges: List[List[int]]
	        :type left: int
	        :type right: int
	        :rtype: bool
	        """
	        a = set(x for x,y in ranges for x in range(x,y+1))
	        b = set(x for x in range(left, right+1))
	        return a.intersection(b)==b

	            	      
			
### 运行结果

	Runtime: 28 ms, faster than 58.93% of Python online submissions for Check if All the Integers in a Range Are Covered.
	Memory Usage: 13.7 MB, less than 7.74% of Python online submissions for Check if All the Integers in a Range Are Covered.


原题链接：https://leetcode.com/problems/check-if-all-the-integers-in-a-range-are-covered



您的支持是我最大的动力
