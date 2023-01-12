leetcode  284. Peeking Iterator（python）




### 描述

Design an iterator that supports the peek operation on an existing iterator in addition to the hasNext and the next operations.

Implement the PeekingIterator class:

* PeekingIterator(Iterator<int> nums) Initializes the object with the given integer iterator iterator.
* int next() Returns the next element in the array and moves the pointer to the next element.
* boolean hasNext() Returns true if there are still elements in the array.
* int peek() Returns the next element in the array without moving the pointer.

Note: Each language may have a different implementation of the constructor and Iterator, but they all support the int next() and boolean hasNext() functions.

 



Example 1:


	Input
	["PeekingIterator", "next", "peek", "next", "next", "hasNext"]
	[[[1, 2, 3]], [], [], [], [], []]
	Output
	[null, 1, 2, 2, 3, false]
	
	Explanation
	PeekingIterator peekingIterator = new PeekingIterator([1, 2, 3]); // [1,2,3]
	peekingIterator.next();    // return 1, the pointer moves to the next element [1,2,3].
	peekingIterator.peek();    // return 2, the pointer does not move [1,2,3].
	peekingIterator.next();    // return 2, the pointer moves to the next element [1,2,3]
	peekingIterator.next();    // return 3, the pointer moves to the next element [1,2,3]
	peekingIterator.hasNext(); // return False
	



Note:

	1 <= nums.length <= 1000
	1 <= nums[i] <= 1000
	All the calls to next and peek are valid.
	At most 1000 calls will be made to next, hasNext, and peek.



### 解析

根据题意，设计一个迭代器，除了 hasNext 和 next 操作之外，还支持对现有迭代器的 peek 操作。

实现 PeekingIterator 类：

* PeekingIterator(Iterator<int> nums) 用给定的整数迭代器 iterator 初始化对象
* int next() 返回数组中的下一个元素并将指针移动到下一个元素
* boolean hasNext() 如果数组中仍有元素，则返回 true
* int peek() 在不移动指针的情况下返回数组中的下一个元素

其实这道题考察的是编程语言中子类操作继承函数，我们在实现 next 和  hasNext 的时候直接调用父类的方法即可，在实现 peek 的时候，因为不能往前移动指针，我们就拷贝一份迭代器 tmp ，将 tmp 的下一个元素返回即可。

每个函数的时间复杂度为 O(1) ，空间复杂度为 O(1) 。


### 解答
				
	class PeekingIterator(object):
	    def __init__(self, iterator):
	        """
	        Initialize your data structure here.
	        :type iterator: Iterator
	        """
	        self.iterator = iterator
	
	    def peek(self):
	        """
	        Returns the next element in the iteration without advancing the iterator.
	        :rtype: int
	        """
	        tmp = copy.deepcopy(self.iterator)
	        return tmp.next()
	        
	        
	
	    def next(self):
	        """
	        :rtype: int
	        """
	        return self.iterator.next()
	        
	
	    def hasNext(self):
	        """
	        :rtype: bool
	        """
	        return self.iterator.hasNext()

            	      
			
### 运行结果

	Runtime: 77 ms, faster than 6.14% of Python online submissions for Peeking Iterator.
	Memory Usage: 13.9 MB, less than 13.16% of Python online submissions for Peeking Iterator.


### 原题链接


https://leetcode.com/problems/peeking-iterator/

您的支持是我最大的动力
