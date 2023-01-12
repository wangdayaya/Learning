leetcode  1944. Number of Visible People in a Queue（python）

### 描述

There are n people standing in a queue, and they numbered from 0 to n - 1 in left to right order. You are given an array heights of distinct integers where heights[i] represents the height of the i<sub>th</sub> person.

A person can see another person to their right in the queue if everybody in between is shorter than both of them. More formally, the i<sub>th</sub> person can see the j<sub>th</sub> person if i < j and min(heights[i], heights[j]) > max(heights[i+1], heights[i+2], ..., heights[j-1]).

Return an array answer of length n where answer[i] is the number of people the i<sub>th</sub> person can see to their right in the queue.



Example 1:

![](https://assets.leetcode.com/uploads/2021/05/29/queue-plane.jpg)

	Input: heights = [10,6,8,5,11,9]
	Output: [3,1,2,1,1,0]
	Explanation:
	Person 0 can see person 1, 2, and 4.
	Person 1 can see person 2.
	Person 2 can see person 3 and 4.
	Person 3 can see person 4.
	Person 4 can see person 5.
	Person 5 can see no one since nobody is to the right of them.

	
Example 2:

	Input: heights = [5,1,2,3,10]
	Output: [4,1,1,1,0]







Note:

	n == heights.length
	1 <= n <= 105
	1 <= heights[i] <= 105
	All the values of heights are unique.



### 解析

根据题意，有 n 个人站在队列中，从 0 到 n - 1 按从左到右的顺序编号。 给定一个由不同整数组成的数组 heights，其中 heights[i] 表示第 i 个人的身高。每个人只能看到右边视线内的人，返回一个长度为 n 的数组 answer，其中 answer[i] 是队列中第 i 个人右边的可见人数。

通过例子一，我们就能发现，每个人  heights[i] 只能看到的是右边身高小于等于他自己的身高，且是递增身高的人，在右侧处于“身高低谷”的人无法被左边的人看到，本质上这就是一道考察单调栈的题目，我们可以从右往左维护一个递减身高序列，当发现有新的身高 h ，就将栈顶比 h 小的身高都弹出（因为这几个人都被 h 遮挡，后边也不会被看到），并计数 count 表示 h 右边可以看到的人，如果此时 stack 不为空则 count+1 ，因为至少还有右边的一个大于等于 h 的身高的人可以被看到。重复这个过程就能找到答案。


### 解答
				
	class Solution(object):
	    def canSeePersonsCount(self, heights):
	        """
	        :type heights: List[int]
	        :rtype: List[int]
	        """
	        N = len(heights)
	        stack = []
	        result = [0] * N
	        for i in range(N-1, -1, -1):
	            count = 0
	            while stack and heights[i]>heights[stack[-1]]:
	                count += 1
	                stack.pop()
	            if stack:
	                count += 1
	            result[i] = count
	            stack.append(i)
	        return result
	            
	            
			
### 运行结果

	Runtime: 1112 ms, faster than 70.59% of Python online submissions for Number of Visible People in a Queue.
	Memory Usage: 26.5 MB, less than 38.24% of Python online submissions for Number of Visible People in a Queue.


原题链接：https://leetcode.com/problems/number-of-visible-people-in-a-queue/



您的支持是我最大的动力
