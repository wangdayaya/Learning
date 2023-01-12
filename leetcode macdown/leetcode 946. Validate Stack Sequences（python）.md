leetcode  946. Validate Stack Sequences（python）




### 描述


Given two integer arrays pushed and popped each with distinct values, return true if this could have been the result of a sequence of push and pop operations on an initially empty stack, or false otherwise.


Example 1:

	Input: pushed = [1,2,3,4,5], popped = [4,5,3,2,1]
	Output: true
	Explanation: We might do the following sequence:
	push(1), push(2), push(3), push(4),
	pop() -> 4,
	push(5),
	pop() -> 5, pop() -> 3, pop() -> 2, pop() -> 1

	





Note:

	1 <= pushed.length <= 1000
	0 <= pushed[i] <= 1000
	All the elements of pushed are unique.
	popped.length == pushed.length
	popped is a permutation of pushed.


### 解析


好像这周的题目都是出的和栈有关系的，也算是一个系列吧，栈的考点是很常见的，正好用这周的题练练手。

根据题意，给定两个具有不同的整数数组 pushed 和 popped ，如果这可能是在最初的一个空栈上，进行的一系列压进和弹出元素的结果，最后又变回空栈，则返回 true，否则返回 false 。

其实说白了就一句话，我们一开始的时候就是空栈，给出了两个列表操作， pushed 和 popped  ，如果按照某一个顺序进行操作，最后仍然是空栈就返回 True 。

首先我们要知道一个常识那就是栈中的元素都是“先进后出”的，如果返回了这个规律那肯定是个不合法的栈，所以在 pushed 中，元素 a 在 b 之前进栈，那么在 popped 中 a 一定是在 b 之后出栈的。

* 我们定义一个空列表 L 当作初始列表，然后定义一个索引 i 用来遍历 pushed ，定义一个索引 j 用来遍历 popped ；
* 在一个 while 循环之中，如果 L 不为空且 L 的栈顶和 popped[j] 相等，那我们就将 L 的栈顶弹出，并且将 j 加一去找下一个需要弹出的元素；否则说明没有需要弹出的元素，我们将 pushed[i] 加入栈顶即可，然后 i 加一去找下一个需要进栈的元素；循环这个过程直到 i 或者 j 越界跳出循环
* 当 j 仍然小于 popped 的长度，说明还有需要弹出的元素，所以如果 popped[j] 等于 L 的栈顶元素就将 L 的栈顶元素弹出，然后 j 加一，直到 j 越界跳出循环
* 最后判断如果 L 为空则返回 True ，如果不为空返回 False 

时间复杂度为 O(N) ，空间复杂度为 O(N) 

### 解答
				

	class Solution(object):
	    def validateStackSequences(self, pushed, popped):
	        L = []
	        i = 0
	        j = 0
	        while i < len(pushed) and j < len(popped):
	            if L and L[-1] == popped[j]:
	                L.pop()
	                j += 1
	            else:
	                L.append(pushed[i])
	                i += 1
	        while j < len(popped):
	            if popped[j] == L[-1]:
	                L.pop()
	            j += 1
	        return True if not L else False
            	      
			
### 运行结果


	Runtime: 60 ms, faster than 78.29% of Python online submissions for Validate Stack Sequences.
	Memory Usage: 13.5 MB, less than 67.76% of Python online submissions for Validate Stack Sequences.


### 原题链接



https://leetcode.com/problems/validate-stack-sequences/


您的支持是我最大的动力
