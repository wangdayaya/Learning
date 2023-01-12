leetcode  2336. Smallest Number in Infinite Set（python）




### 描述

You have a set which contains all positive integers [1, 2, 3, 4, 5, ...]. Implement the SmallestInfiniteSet class:

* SmallestInfiniteSet() Initializes the SmallestInfiniteSet object to contain all positive integers.
* int popSmallest() Removes and returns the smallest integer contained in the infinite set.
* void addBack(int num) Adds a positive integer num back into the infinite set, if it is not already in the infinite set.



Example 1:

	Input
	["SmallestInfiniteSet", "addBack", "popSmallest", "popSmallest", "popSmallest", "addBack", "popSmallest", "popSmallest", "popSmallest"]
	[[], [2], [], [], [], [1], [], [], []]
	Output
	[null, null, 1, 2, 3, null, 1, 4, 5]
	
	Explanation
	SmallestInfiniteSet smallestInfiniteSet = new SmallestInfiniteSet();
	smallestInfiniteSet.addBack(2);    // 2 is already in the set, so no change is made.
	smallestInfiniteSet.popSmallest(); // return 1, since 1 is the smallest number, and remove it from the set.
	smallestInfiniteSet.popSmallest(); // return 2, and remove it from the set.
	smallestInfiniteSet.popSmallest(); // return 3, and remove it from the set.
	smallestInfiniteSet.addBack(1);    // 1 is added back to the set.
	smallestInfiniteSet.popSmallest(); // return 1, since 1 was added back to the set and
	                                   // is the smallest number, and remove it from the set.
	smallestInfiniteSet.popSmallest(); // return 4, and remove it from the set.
	smallestInfiniteSet.popSmallest(); // return 5, and remove it from the set.






Note:

	1 <= num <= 1000
	At most 1000 calls will be made in total to popSmallest and addBack.


### 解析

根据题意，给定一个包含所有正整数 [1, 2, 3, 4, 5, ...] 的集合。 实现 SmallestInfiniteSet 类：

* SmallestInfiniteSet() 初始化 SmallestInfiniteSet 对象以包含所有正整数。
* int popSmallest() 删除并返回无限集中包含的最小整数。
* void addBack(int num) 如果它尚未在无限集中，将正整数 num 添加回无限集中。

其实这道题很简单，因为限制条件中 num 的数字最大为 1000 ，而且最多操作 1000 次函数，而且包含的是从 1 到 n 的所有正整数集合，所以我们用到了堆数据结构，只需要初始化的时候，将 1 到 1000 的正整数放入小根堆，然后当执行 popSmallest 的时候只需要弹出最小的整数即可，当执行 addBack(int num) 的时候，如果  num 不在小根堆，就将其加入即可。

时间复杂度为 O(NlogN) ，空间复杂度为 O(N) 。


### 解答

	class SmallestInfiniteSet(object):
	
	    def __init__(self):
	        self.L = [i for i in range(1,1001)]
	        heapq.heapify(self.L)
	
	    def popSmallest(self):
	        """
	        :rtype: int
	        """
	        if self.L:
	            return heapq.heappop(self.L)
	
	    def addBack(self, num):
	        """
	        :type num: int
	        :rtype: None
	        """
	        if num not in self.L:
	            heapq.heappush(self.L, num)

### 运行结果

		
	135 / 135 test cases passed.
	Status: Accepted
	Runtime: 455 ms
	Memory Usage: 14.2 MB
### 原题链接

	https://leetcode.com/contest/weekly-contest-301/problems/smallest-number-in-infinite-set/


您的支持是我最大的动力
