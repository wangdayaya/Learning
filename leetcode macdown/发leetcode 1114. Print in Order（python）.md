leetcode  1114. Print in Order（python）

### 描述

Suppose we have a class:

	public class Foo {
	  public void first() { print("first"); }
	  public void second() { print("second"); }
	  public void third() { print("third"); }
	}

The same instance of Foo will be passed to three different threads. Thread A will call first(), thread B will call second(), and thread C will call third(). Design a mechanism and modify the program to ensure that second() is executed after first(), and third() is executed after second().

Note:

We do not know how the threads will be scheduled in the operating system, even though the numbers in the input seem to imply the ordering. The input format you see is mainly to ensure our tests' comprehensiveness.



Example 1:

	Input: nums = [1,2,3]
	Output: "firstsecondthird"
	Explanation: There are three threads being fired asynchronously. The input [1,2,3] means thread A calls first(), thread B calls second(), and thread C calls third(). "firstsecondthird" is the correct output.

	
Example 2:


	Input: nums = [1,3,2]
	Output: "firstsecondthird"
	Explanation: The input [1,3,2] means thread A calls first(), thread B calls third(), and thread C calls second(). "firstsecondthird" is the correct output.





### 解析

根据题意，就是要在给定 nums 顺序的情况下，保证 second() 在 first() 之后执行，third() 在  second()  之后执行，其实就是在考 python 线程的知识点，关键是获取锁和释放锁。

在 python 中，Lock 是目前可用的最低级的同步原语，实现自 _thread 扩展模块。

* 原语锁有两种状态：locked (锁定)或 unlocked (未锁定)。创建时为未锁定状态。
* 原语锁有两种方法：acquire() 和 release() 。当锁处于未锁定状态时， acquire() 改变其为锁定状态。当锁处于锁定状态时，调用 acquire() 方法将导致线程阻塞，直到其他线程调用 release() 释放锁。

题中要保证 second() 在 first() 之后执行，third() 在  second()  之后执行，所以需要两把锁，第一把锁控制  second() 在 first() 之后执行 ，第二把锁控制 third() 在  second()  之后执行。

### 解答
				
	class Foo(object):
	    def __init__(self):
	        self.lock1 = threading.Lock()
	        self.lock1.acquire()
	
	        self.lock2 = threading.Lock()
	        self.lock2.acquire()
	
	
	    def first(self, printFirst):
	        """
	        :type printFirst: method
	        :rtype: void
	        """
	        printFirst()
	        self.lock1.release()
	
	    def second(self, printSecond):
	        """
	        :type printSecond: method
	        :rtype: void
	        """
	        self.lock1.acquire()
	        printSecond()
	        self.lock2.release()
	
	    def third(self, printThird):
	        """
	        :type printThird: method
	        :rtype: void
	        """
	        self.lock2.acquire()
	        printThird()
	


            	      
			
### 运行结果


	Runtime: 32 ms, faster than 60.98% of Python online submissions for Print in Order.
	Memory Usage: 13.6 MB, less than 59.15% of Python online submissions for Print in Order.

原题链接：https://leetcode.com/problems/print-in-order



您的支持是我最大的动力
