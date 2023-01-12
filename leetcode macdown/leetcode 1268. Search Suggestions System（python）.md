leetcode  （python）




### 描述

You are given an array of strings products and a string searchWord.

Design a system that suggests at most three product names from products after each character of searchWord is typed. Suggested products should have common prefix with searchWord. If there are more than three products with a common prefix return the three lexicographically minimums products.

Return a list of lists of the suggested products after each character of searchWord is typed.



Example 1:

	Input: products = ["mobile","mouse","moneypot","monitor","mousepad"], searchWord = "mouse"
	Output: [
	["mobile","moneypot","monitor"],
	["mobile","moneypot","monitor"],
	["mouse","mousepad"],
	["mouse","mousepad"],
	["mouse","mousepad"]
	]
	Explanation: products sorted lexicographically = ["mobile","moneypot","monitor","mouse","mousepad"]
	After typing m and mo all products match and we show user ["mobile","moneypot","monitor"]
	After typing mou, mous and mouse the system suggests ["mouse","mousepad"]

	
Example 2:

	Input: products = ["havana"], searchWord = "havana"
	Output: [["havana"],["havana"],["havana"],["havana"],["havana"],["havana"]]


Example 3:


	Input: products = ["bags","baggage","banner","box","cloths"], searchWord = "bags"
	Output: [["baggage","bags","banner"],["baggage","bags","banner"],["baggage","bags"],["bags"]]
	



Note:


	1 <= products.length <= 1000
	1 <= products[i].length <= 3000
	1 <= sum(products[i].length) <= 2 * 10^4
	All the strings of products are unique.
	products[i] consists of lowercase English letters.
	1 <= searchWord.length <= 1000
	searchWord consists of lowercase English letters.

### 解析

根据题意，给定一个字符串 products 数组和一个字符串 searchWord 。

设计一个系统，在输入 searchWord 的每个字符后，从产品中最多给出三个建议的产品名称。 建议的产品应该与 searchWord 有共同的前缀。 如果有超过三个具有公共前缀的产品，则返回三个按字典顺序排列的产品名字。在输入 searchWord 的每个字符后返回建议产品列表的列表。

通过看限制条件我们知道，因为输入长度在 1000 以内，可以使用暴力结局该题，直接将所有的前缀找出来，然后取不同的前缀在 products 中找符合要求的最多 3 个产品将其放入结果列表 result 中，最后返回 result 即可。

时间复杂度为 O(N\*M) ，N 为 searchWord长度， M 为 products 长度，空间复杂度最大为 O(3\*N)，N 为 searchWord 长度。

### 解答
				

	class Solution(object):
	    def suggestedProducts(self, products, searchWord):
	        """
	        :type products: List[str]
	        :type searchWord: str
	        :rtype: List[List[str]]
	        """
	        products.sort()
	        N = len(searchWord)
	        result = []
	        for i in range(1,N+1):
	            pre = searchWord[:i]
	            tmp = []
	            for p in products:
	                if p.startswith(pre):
	                    tmp.append(p)
	                    if len(tmp) == 3:
	                        result.append(tmp)
	                        break
	            if len(tmp)<3:
	                result.append(tmp)
	        return result
            	      
			
### 运行结果

	Runtime: 336 ms, faster than 43.73% of Python online submissions for Search Suggestions System.
	Memory Usage: 15.6 MB, less than 50.33% of Python online submissions for Search Suggestions System.


### 解析

暴力解法固然可以 AC ，但是没有什么技术含量，我们其实拿到某个 searchWord 的某个前缀 pre 的时候，只需要通过二分法去有序的 products 中去定位该前缀的位置，然后找出其后面紧挨的最多 3 个合法字符串，将其将入到 result 中即可。

时间复杂度为 O(NlogM) ，N 为 searchWord长度， M 为 products 长度，空间复杂度为 O(3\*N) ，N 为 searchWord 长度。


### 解答

	class Solution(object):
	    def suggestedProducts(self, products, searchWord):
	        """
	        :type products: List[str]
	        :type searchWord: str
	        :rtype: List[List[str]]
	        """
	        products.sort()
	        result = []
	        N = len(searchWord)
	        for i in range(1, N + 1):
	            pre = searchWord[:i]
	            idx = bisect.bisect_left(products, pre)
	            tmp = [products[j] for j in range(idx, min(len(products), idx + 3)) if products[j].startswith(pre)]
	            result.append(tmp)
	        return result



### 运行结果

	Runtime: 64 ms, faster than 97.80% of Python online submissions for Search Suggestions System.
	Memory Usage: 15.5 MB, less than 67.91% of Python online submissions for Search Suggestions System.

### 解析

其实对于这种找前缀相同的字符串，还有一种数据结构很适合，那就是 Trie 树 ，我们可以根据题意，构造一个 Trie 树， 为每一种前缀找出按照字典顺序排列的最多 3 个产品名称，最后构造完成之后，根结点到某个叶子结点形成的字符串即为 prefix ，直接将当前结点中的包含最多 3 个产品名称的列表加入 result 中即可。

时间复杂度为 O(N+S) ，N 为所有 products 的字符数量，S 为 searchWord 的字符数量，空间复杂度为 O(3\*N) ，N 为树的结点数量，每个节点中最多放 3 个元素。

### 解答

	class Trie:
	    def __init__(self):
	        self.child = dict()
	        self.words = list()
	        
	class Solution(object):
	    def suggestedProducts(self, products, searchWord):
	        def add(node, word):
	            cur = node
	            for c in word:
	                if c not in cur.child:
	                    cur.child[c] = Trie()
	                cur = cur.child[c]
	                cur.words.append(word)
	                cur.words.sort()
	                if len(cur.words) > 3:
	                    cur.words.pop()
	        root = Trie()
	        for word in products:
	            add(root, word)
	
	        result = []
	        cur = root
	        flag = False
	        for c in searchWord:
	            if flag or c not in cur.child:
	                result.append([])
	                flag = True
	            else:
	                cur = cur.child[c]
	                result.append(cur.words)
	        return result

### 运行结果


### 原题链接


https://leetcode.com/problems/search-suggestions-system/


您的支持是我最大的动力
