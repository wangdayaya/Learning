leetcode  2349. Design a Number Container System（python）




### 描述



Design a number container system that can do the following:

* Insert or Replace a number at the given index in the system.
* Return the smallest index for the given number in the system.

Implement the NumberContainers class:

* NumberContainers() Initializes the number container system.
* void change(int index, int number) Fills the container at index with the number. If there is already a number at that index, replace it.
* int find(int number) Returns the smallest index for the given number, or -1 if there is no index that is filled by number in the system.

Example 1:


	Input
	["NumberContainers", "find", "change", "change", "change", "change", "find", "change", "find"]
	[[], [10], [2, 10], [1, 10], [3, 10], [5, 10], [10], [1, 20], [10]]
	Output
	[null, -1, null, null, null, null, 1, null, 2]
	
	Explanation
	NumberContainers nc = new NumberContainers();
	nc.find(10); // There is no index that is filled with number 10. Therefore, we return -1.
	nc.change(2, 10); // Your container at index 2 will be filled with number 10.
	nc.change(1, 10); // Your container at index 1 will be filled with number 10.
	nc.change(3, 10); // Your container at index 3 will be filled with number 10.
	nc.change(5, 10); // Your container at index 5 will be filled with number 10.
	nc.find(10); // Number 10 is at the indices 1, 2, 3, and 5. Since the smallest index that is filled with 10 is 1, we return 1.
	nc.change(1, 20); // Your container at index 1 will be filled with number 20. Note that index 1 was filled with 10 and then replaced with 20. 
	nc.find(10); // Number 10 is at the indices 2, 3, and 5. The smallest index that is filled with 10 is 2. Therefore, we return 2.
	






Note:

	1 <= index, number <= 109
	At most 10^5 calls will be made in total to change and find.


### 解析

根据题意，设计一个可以执行以下操作的数字容器系统：

* 在系统中的给定索引处插入或替换一个数字。
* 返回系统中给定数字的最小索引。

实现 NumberContainers 类：

* NumberContainers() 初始化数字容器系统。
* void change(int index, int number) 用数字填充索引处的容器。 如果该索引处已经有一个数字，请将其替换。
* int find(int number) 返回给定数字的最小索引，如果系统中没有由数字填充的索引，则返回 -1。

这道题的关键就在于 change 和 find 的方法要调用 10^5 ，所以每个函数的时间复杂度都必须要在 O(NlogN) 以下，分别对两个函数进行解析：

而 find 要找出某个数字的最小索引，这就需要我们事先保存好该数字对应的索引集合，而且为了最快取出最小的索引，这个集合应该是有序集合，所以我们用字典结合有序集合保存某个数字对应的索引集合，定义为 n2s ，这样我们将 find 的时间复杂度降到了 O(1) ，空间复杂度为 O(1) 。

change 要对某个索引的值进行改变，自然用字典进行填充和修改是最快的，定义为 i2n ；但是因为要改变之前索引对应的值 old ，所以要将 n2s[old] 对应的索引删除，然后更新 i2n[index] 为 number ，n2s[number] 将最新的索引收录，因为每次加入新的索引要进行排序，所以时间复杂度主要消耗在了有序集合的排序中，时间复杂度为 O(nlogN) ，空间复杂度为 O(N) 。


### 解答

	from sortedcontainers import  SortedSet
	class NumberContainers(object):
	
	    def __init__(self):
	        self.i2n = collections.defaultdict()
	        self.n2s = collections.defaultdict(SortedSet)
	
	    def change(self, index, number):
	        if index in self.i2n:
	            n = self.i2n[index]
	            self.n2s[n].remove(index)
	        self.i2n[index] = number
	        self.n2s[number].add(index)
	
	    def find(self, number):
	        if number in self.n2s and len(self.n2s[number]) > 0:
	            return self.n2s[number][0]
	        return -1

### 运行结果

	
	44 / 44 test cases passed.
	Status: Accepted
	Runtime: 2168 ms
	Memory Usage: 62.1 MB

### 原题链接

https://leetcode.com/contest/biweekly-contest-83/problems/design-a-number-container-system/


您的支持是我最大的动力
