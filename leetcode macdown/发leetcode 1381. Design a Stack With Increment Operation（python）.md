leetcode  1381. Design a Stack With Increment Operation（python）

### 描述

Design a stack which supports the following operations.

Implement the CustomStack class:

* CustomStack(int maxSize) Initializes the object with maxSize which is the maximum number of elements in the stack or do nothing if the stack reached the maxSize.
* void push(int x) Adds x to the top of the stack if the stack hasn't reached the maxSize.
* int pop() Pops and returns the top of stack or -1 if the stack is empty.
* void inc(int k, int val) Increments the bottom k elements of the stack by val. If there are less than k elements in the stack, just increment all the elements in the stack.




Example 1:


	Input
	["CustomStack","push","push","pop","push","push","push","increment","increment","pop","pop","pop","pop"]
	[[3],[1],[2],[],[2],[3],[4],[5,100],[2,100],[],[],[],[]]
	Output
	[null,null,null,2,null,null,null,null,null,103,202,201,-1]
	Explanation
	CustomStack customStack = new CustomStack(3); // Stack is Empty []
	customStack.push(1);                          // stack becomes [1]
	customStack.push(2);                          // stack becomes [1, 2]
	customStack.pop();                            // return 2 --> Return top of the stack 2, stack becomes [1]
	customStack.push(2);                          // stack becomes [1, 2]
	customStack.push(3);                          // stack becomes [1, 2, 3]
	customStack.push(4);                          // stack still [1, 2, 3], Don't add another elements as size is 4
	customStack.increment(5, 100);                // stack becomes [101, 102, 103]
	customStack.increment(2, 100);                // stack becomes [201, 202, 103]
	customStack.pop();                            // return 103 --> Return top of the stack 103, stack becomes [201, 202]
	customStack.pop();                            // return 202 --> Return top of the stack 102, stack becomes [201]
	customStack.pop();                            // return 201 --> Return top of the stack 101, stack becomes []
	customStack.pop();                            // return -1 --> Stack is empty return -1.


Note:


	1 <= maxSize <= 1000
	1 <= x <= 1000
	1 <= k <= 1000
	0 <= val <= 100
	At most 1000 calls will be made to each method of increment, push and pop each separately.

### 解析

根据题意，就是要求自己设计一个栈相关的操作，每个方法都规定了输入、操作方法和输出，思路比较简单：通过列表及其相关的内置函数完成相关的操作即可，只要懂得了栈的概念，实现起来比较简单。

* \_\_init__ 函数中主要是初始化列表 self.stack 和 self.maxSize 两个全局变量
* push 函数主要是在 self.stack 长度不大于 self.maxSize 的情况下将元素压入栈中
* pop 函数主要是当 self.stack 不为空的时候返回栈顶的元素，否则返回 -1
* increment 函数就是将底部的 k 个元素都加上 val ，如果 self.stack 长度小于 k ，则全栈元素都加上 val



### 解答
				

	class CustomStack(object):
	
	    def __init__(self, maxSize):
	        """
	        :type maxSize: int
	        """
	        self.stack = []
	        self.maxSize = maxSize
	
	    def push(self, x):
	        """
	        :type x: int
	        :rtype: None
	        """
	        if len(self.stack)<self.maxSize:
	            self.stack.append(x)
	
	
	    def pop(self):
	        """
	        :rtype: int
	        """
	        if len(self.stack)>0:
	            return self.stack.pop()
	        return -1
	
	    def increment(self, k, val):
	        """
	        :type k: int
	        :type val: int
	        :rtype: None
	        """
	        if self.stack:
	            for i,v  in enumerate(self.stack[:k]):
	                self.stack[i] += val
	        
	            	      
			
### 运行结果

	Runtime: 132 ms, faster than 27.21% of Python online submissions for Design a Stack With Increment Operation.
	Memory Usage: 14.5 MB, less than 9.56% of Python online submissions for Design a Stack With Increment Operation.



原题链接：https://leetcode.com/problems/design-a-stack-with-increment-operation/



您的支持是我最大的动力
