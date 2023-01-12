leetcode  705. Design HashSet（python）

### 描述

Design a HashSet without using any built-in hash table libraries.

Implement MyHashSet class:

* void add(key) Inserts the value key into the HashSet.
* bool contains(key) Returns whether the value key exists in the HashSet or not.
* void remove(key) Removes the value key in the HashSet. If key does not exist in the HashSet, do nothing.




Example 1:

	Input
	["MyHashSet", "add", "add", "contains", "contains", "add", "contains", "remove", "contains"]
	[[], [1], [2], [1], [3], [2], [2], [2], [2]]
	Output
	[null, null, null, true, false, null, true, null, false]
	
	Explanation
	MyHashSet myHashSet = new MyHashSet();
	myHashSet.add(1);      // set = [1]
	myHashSet.add(2);      // set = [1, 2]
	myHashSet.contains(1); // return True
	myHashSet.contains(3); // return False, (not found)
	myHashSet.add(2);      // set = [1, 2]
	myHashSet.contains(2); // return True
	myHashSet.remove(2);   // set = [1]
	myHashSet.contains(2); // return False, (already removed)

	


Note:

	0 <= key <= 10^6
	At most 104 calls will be made to add, remove, and contains.



### 解析


根据题意，就是实现一个 Hashset ，直接使用列表的相关操作即可实现，很简单，不再赘述。这种方法使用了内置函数，不推荐使用。

### 解答
				

	class MyHashSet(object):
	
	    def __init__(self):
	        """
	        Initialize your data structure here.
	        """
	        self.l = []
	    def add(self, key):
	        """
	        :type key: int
	        :rtype: None
	        """
	        if key not in self.l:
	            self.l.append(key)
	
	    def remove(self, key):
	        """
	        :type key: int
	        :rtype: None
	        """
	        if key in self.l:
	            self.l.remove(key)
	        
	    def contains(self, key):
	        """
	        Returns true if this set contains the specified element
	        :type key: int
	        :rtype: bool
	        """
	        if key in self.l:
	            return True
	        return False
	
	

            	      
			
### 运行结果

	
	Runtime: 1104 ms, faster than 25.11% of Python online submissions for Design HashSet.
	Memory Usage: 18.3 MB, less than 85.65% of Python online submissions for Design HashSet.
### 解析


另外一种思路就是初始化一个数组，因为上面限制了 key 的长度在 10^6 之内，所以数组的长度设置成 1000001 ，全部都设置为 False ，如果有数字就设置为 True ，利用数组的基本操作完成题目中的各个函数功能。

### 解答
	class MyHashSet(object):
	
	    def __init__(self):
	        """
	        Initialize your data structure here.
	        """
	        self.set = [False] * 1000001
	    def add(self, key):
	        """
	        :type key: int
	        :rtype: None
	        """
	        self.set[key] = True
	
	    def remove(self, key):
	        """
	        :type key: int
	        :rtype: None
	        """
	        self.set[key] = False
	        
	    def contains(self, key):
	        """
	        Returns true if this set contains the specified element
	        :type key: int
	        :rtype: bool
	        """
	        return self.set[key]
	


### 运行结果

	Runtime: 356 ms, faster than 35.60% of Python online submissions for Design HashSet.
	Memory Usage: 41.2 MB, less than 5.18% of Python online submissions for Design HashSet.

原题链接：https://leetcode.com/problems/design-hashset/



您的支持是我最大的动力
