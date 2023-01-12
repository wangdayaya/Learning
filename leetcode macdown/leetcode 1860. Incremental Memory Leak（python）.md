leetcode  1860. Incremental Memory Leak（python）

### 描述

You are given two integers memory1 and memory2 representing the available memory in bits on two memory sticks. There is currently a faulty program running that consumes an increasing amount of memory every second.

At the i<sup>th</sup> second (starting from 1), i bits of memory are allocated to the stick with more available memory (or from the first memory stick if both have the same available memory). If neither stick has at least i bits of available memory, the program crashes.

Return an array containing [crashTime, memory1<sub>crash</sub>, memory2<sub>crash</sub>], where crashTime is the time (in seconds) when the program crashed and memory1<sub>crash</sub> and memory2<sub>crash</sub> are the available bits of memory in the first and second sticks respectively.





Example 1:

	Input: memory1 = 2, memory2 = 2
	Output: [3,1,0]
	Explanation: The memory is allocated as follows:
	- At the 1st second, 1 bit of memory is allocated to stick 1. The first stick now has 1 bit of available memory.
	- At the 2nd second, 2 bits of memory are allocated to stick 2. The second stick now has 0 bits of available memory.
	- At the 3rd second, the program crashes. The sticks have 1 and 0 bits available respectively.

	
Example 2:

	Input: memory1 = 8, memory2 = 11
	Output: [6,0,4]
	Explanation: The memory is allocated as follows:
	- At the 1st second, 1 bit of memory is allocated to stick 2. The second stick now has 10 bit of available memory.
	- At the 2nd second, 2 bits of memory are allocated to stick 2. The second stick now has 8 bits of available memory.
	- At the 3rd second, 3 bits of memory are allocated to stick 1. The first stick now has 5 bits of available memory.
	- At the 4th second, 4 bits of memory are allocated to stick 2. The second stick now has 4 bits of available memory.
	- At the 5th second, 5 bits of memory are allocated to stick 1. The first stick now has 0 bits of available memory.
	- At the 6th second, the program crashes. The sticks have 0 and 4 bits available respectively.





Note:

	0 <= memory1, memory2 <= 2^31 - 1



### 解析

根据题意， 给出两个整数 memory1 和 memory2，分别表示两个内存条上的可用内存位数。 当前有一个错误的程序正在运行，每秒消耗的内存量越来越大。在第 i 秒（从 1 开始）需要在有更多空间的内存条上划分 i 位的内存空间（如果两个内存条有相同的可用内存，则从第一个内存条中划分空间）。 如果两个内存条都没有至少有 i 位可用内存，程序就会崩溃。

返回一个包含 [crashTime, memory1crash, memory2crash] 的数组，其中 crashTime 是程序崩溃的时间（以秒为单位），而 memory1crash 和 memory2crash 分别是第一个和第二个内存条中剩下的可用内存位数。

这个题很简单，就直接模仿题意进行写代码即可，如果两个内存条都有可供第 i 秒使用的大小为 i 位的空间，那就去某一个内存条上划分空间。关键就在于每次划分内存在有更多空间的内存条上进行；如果两个内存条上剩余的空间相等，在第一个内存条上进行。


### 解答
				
	class Solution(object):
	    def memLeak(self, memory1, memory2):
	        """
	        :type memory1: int
	        :type memory2: int
	        :rtype: List[int]
	        """
	        i = 1
	        while max(memory1, memory2)>=i:
	            if memory1>=memory2:
	                memory1 -= i
	            else:
	                memory2 -= i
	            i += 1
	        return [i, memory1, memory2]

            	      
			
### 运行结果

	
	Runtime: 424 ms, faster than 40.00% of Python online submissions for Incremental Memory Leak.
	Memory Usage: 13.4 MB, less than 76.67% of Python online submissions for Incremental Memory Leak.


原题链接：https://leetcode.com/problems/incremental-memory-leak/



您的支持是我最大的动力
