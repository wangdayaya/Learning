leetcode 706. Design HashMap （python）

### 描述

Design a HashMap without using any built-in hash table libraries.

Implement the MyHashMap class:

* MyHashMap() initializes the object with an empty map.
* void put(int key, int value) inserts a (key, value) pair into the HashMap. If the key already exists in the map, update the corresponding value.
* int get(int key) returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key.
* void remove(key) removes the key and its corresponding value if the map contains the mapping for the key.



Example 1:

	Input
	["MyHashMap", "put", "put", "get", "get", "put", "get", "remove", "get"]
	[[], [1, 1], [2, 2], [1], [3], [2, 1], [2], [2], [2]]
	Output
	[null, null, null, 1, -1, null, 1, null, -1]
	
	Explanation
	MyHashMap myHashMap = new MyHashMap();
	myHashMap.put(1, 1); // The map is now [[1,1]]
	myHashMap.put(2, 2); // The map is now [[1,1], [2,2]]
	myHashMap.get(1);    // return 1, The map is now [[1,1], [2,2]]
	myHashMap.get(3);    // return -1 (i.e., not found), The map is now [[1,1], [2,2]]
	myHashMap.put(2, 1); // The map is now [[1,1], [2,1]] (i.e., update the existing value)
	myHashMap.get(2);    // return 1, The map is now [[1,1], [2,1]]
	myHashMap.remove(2); // remove the mapping for 2, The map is now [[1,1]]
	myHashMap.get(2);    // return -1 (i.e., not found), The map is now [[1,1]]

	


Note:

	0 <= key, value <= 106
	At most 104 calls will be made to put, get, and remove.


### 解析


根据题意，就是实现题目中所描述的 hashmap 相关的操作，借助 dict 的相关操作可以完成，不再赘述，这种解法虽然简单，但是有使用内置函数的嫌疑。

### 解答
				


	 class MyHashMap(object):
	
	    def __init__(self):
	        """
	        Initialize your data structure here.
	        """
	        self.d = {}
	
	    def put(self, key, value):
	        """
	        value will always be non-negative.
	        :type key: int
	        :type value: int
	        :rtype: None
	        """
	        self.d[key] = value
	
	    def get(self, key):
	        """
	        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
	        :type key: int
	        :rtype: int
	        """
	        if key in self.d:
	            return self.d[key]
	        return -1
	
	    def remove(self, key):
	        """
	        Removes the mapping of the specified value key if this map contains a mapping for the key
	        :type key: int
	        :rtype: None
	        """
	        if key in self.d:
	            self.d.pop(key)           	      
			
### 运行结果

	
	Runtime: 200 ms, faster than 88.65% of Python online submissions for Design HashMap.
	Memory Usage: 16.3 MB, less than 95.14% of Python online submissions for Design HashMap.

### 解析

另外，可以借用列表来完成相关的操作，将列表中的某个索引当作 key ，将值当作 value ，然后进行相关的操作即可。因为题目中限制了 key 的大小，所以初始化的时候将列表长度设置为 1000001 。

### 解答


	class MyHashMap(object):
	
	    def __init__(self):
	        """
	        Initialize your data structure here.
	        """
	        self.data = [None] * 1000001
	
	    def put(self, key, value):
	        """
	        value will always be non-negative.
	        :type key: int
	        :type value: int
	        :rtype: None
	        """
	        self.data[key] = value
	
	    def get(self, key):
	        """
	        Returns the value to which the specified key is mapped, or -1 if this map contains no mapping for the key
	        :type key: int
	        :rtype: int
	        """
	        value = self.data[key]
	        return value if value != None else -1
	
	    def remove(self, key):
	        """
	        Removes the mapping of the specified value key if this map contains a mapping for the key
	        :type key: int
	        :rtype: None
	        """
	        self.data[key] = None
	        
	
	
	# Your MyHashMap object will be instantiated and called as such:
	# obj = MyHashMap()
	# obj.put(key,value)
	# param_2 = obj.get(key)
	# obj.remove(key)
	
### 运行结果

	Runtime: 368 ms, faster than 38.96% of Python online submissions for Design HashMap.
	Memory Usage: 38 MB, less than 12.01% of Python online submissions for Design HashMap.

### 解析
还有一种是使用 Hash 原理来设计 HashMap ，这种比较复杂用到了链表的原理，这个我没有想到，想进一步了解的可以查看官方解法。

原题链接：https://leetcode.com/problems/design-hashmap/



您的支持是我最大的动力
