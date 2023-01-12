leetcode  1286. Iterator for Combination（python）

### 描述


Design the CombinationIterator class:

* CombinationIterator(string characters, int combinationLength) Initializes the object with a string characters of sorted distinct lowercase English letters and a number combinationLength as arguments.
* next() Returns the next combination of length combinationLength in lexicographical order.
* hasNext() Returns true if and only if there exists a next combination.



Example 1:

	Input
	["CombinationIterator", "next", "hasNext", "next", "hasNext", "next", "hasNext"]
	[["abc", 2], [], [], [], [], [], []]
	Output
	[null, "ab", true, "ac", true, "bc", false]
	
	Explanation
	CombinationIterator itr = new CombinationIterator("abc", 2);
	itr.next();    // return "ab"
	itr.hasNext(); // return True
	itr.next();    // return "ac"
	itr.hasNext(); // return True
	itr.next();    // return "bc"
	itr.hasNext(); // return False
	



Note:

	1 <= combinationLength <= characters.length <= 15
	All the characters of characters are unique.
	At most 10^4 calls will be made to next and hasNext.
	It's guaranteed that all calls of the function next are valid.


### 解析

根据题意，设计一个 CombinationIterator 类，里面需要包涵几个函数：

* CombinationIterator(string characters, int combineLength) 使用已经经过排序的不同小写英文字母的字符串 characters 和数字 combinationLength 作为参数初始化对象
* next() 按字典顺序返回长度为 combinationLength 的下一个组合
* hasNext() 当且仅当存在下一个组合时才返回 true

这道题的关键在于找出所有的组合，至于 next 和 hasNext 两个函数只需要做简单的逻辑判断即可，第一种方法使用了内置函数 itertools.combinations 来找出所有长度为 combinationLength 的组合，简单粗暴。

### 解答
				
	class CombinationIterator(object):
	
	    def __init__(self, characters, combinationLength):
	        """
	        :type characters: str
	        :type combinationLength: int
	        """
	        self.characters = characters
	        self.L = [''.join(i) for i in itertools.combinations(characters, combinationLength)]
	        
	
	    def next(self):
	        """
	        :rtype: str
	        """
	        return self.L.pop(0)
	        
	        
	    def hasNext(self):
	        """
	        :rtype: bool
	        """
	        if self.L:
	            return True
	        return False
	        

            	      
			
### 运行结果

	Runtime: 44 ms, faster than 81.82% of Python online submissions for Iterator for Combination.
	Memory Usage: 16 MB, less than 72.73% of Python online submissions for Iterator for Combination.


### 解析

当然了使用内置函数显得没有水平，我们还可以自己写代码，自定义函数 permute 使用 DFS 找出所有的组合。

### 解答
	
	class CombinationIterator(object):
	
	    def __init__(self, characters, combinationLength):
	        """
	        :type characters: str
	        :type combinationLength: int
	        """
	        self.characters = characters
	        self.n = combinationLength
	        self.N = len(characters)
	        self.L = []
	        self.permute('', 0)
	    
	    def permute(self, s, start):
	        if len(s) == self.n:
	            self.L.append(s)
	            return
	        else:
	            for i in range(start, self.N):
	                self.permute(s + self.characters[i], i + 1)
	
	    def next(self):
	        """
	        :rtype: str
	        """
	        return self.L.pop(0)
	        
	        
	    def hasNext(self):
	        """
	        :rtype: bool
	        """
	        if self.L:
	            return True
	        return False


### 运行结果

	Runtime: 52 ms, faster than 54.55% of Python online submissions for Iterator for Combination.
	Memory Usage: 15.9 MB, less than 72.73% of Python online submissions for Iterator for Combination.

原题链接：https://leetcode.com/problems/iterator-for-combination/



您的支持是我最大的动力
