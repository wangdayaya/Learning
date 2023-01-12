leetcode  520. Detect Capital（python）




### 描述

We define the usage of capitals in a word to be right when one of the following cases holds:

* All letters in this word are capitals, like "USA".
* All letters in this word are not capitals, like "leetcode".
* Only the first letter in this word is capital, like "Google".

Given a string word, return true if the usage of capitals in it is right.




Example 1:

	Input: word = "USA"
	Output: true

	
Example 2:


	Input: word = "FlaG"
	Output: false




Note:

	1 <= word.length <= 100
	word consists of lowercase and uppercase English letters.


### 解析


根据题意，当以下情况之一成立时，我们将单词中大写的使用方法定义为正确：

* 这个词中的所有字母都是大写的，比如 “USA” 。
* 这个词中的所有字母都不是大写的，比如 “leetcode” 。
* 只有这个单词的第一个字母是大写的，比如 “Google” 。

给定一个字符串 word ，如果其中的大写字母使用正确，则返回 true。这道题考查的就是最基础的字符串大小写判断，其实用 python 的内置函数很简单：

* 字符串.islower() ，所有字符都是小写，为真返回 Ture，否则返回 False
* 字符串.isupper() ，所有字符都是大写，为真返回 Ture，否则返回 False
* 字符串.istitle() ，所有单词都是首字母大写，为真返回 Ture，否则返回 False

所以只需要一行代码，只需要判断 word.islower() or word.isupper()  or word.istitle()  为 True 即可。



### 解答
				

	class Solution(object):
	    def detectCapitalUse(self, word):
	        """
	        :type word: str
	        :rtype: bool
	        """
	        return word.islower() or word.isupper()  or word.istitle() 
            	      
			
### 运行结果
	Runtime: 12 ms, faster than 96.89% of Python online submissions for Detect Capital.
	Memory Usage: 13.3 MB, less than 83.94% of Python online submissions for Detect Capital.

### 解析

上面用内置函数尽管方便，但是我还是要自己实现一下相关的算法，就当是为了练手，思路也很简单，就是自己写三个函数，分别判断这三种情况，如果有任意一个为 True ，那么就返回 True 。其实细想一下只要两个函数 self.islowerOrTitle(word) or self.isupper(word)   为 True 就可以了：

* islowerOrTitle ：不管第一个是大写字母还是小写字母，只要从第二位开始后面的字符都是小写字符就可以判断为 True ，符合题目中的第二种和第三种的情况
* isupper ：只要所有的字母都是大写字母就可以判断为 True ，符合题目中第一种的情况

### 解答
	class Solution(object):
	    def detectCapitalUse(self, word):
	        """
	        :type word: str
	        :rtype: bool
	        """
	        return self.islowerOrTitle(word) or self.isupper(word)  
	    
	    def islowerOrTitle(self, s):
	        for c in s[1:]:
	            if not 97<=ord(c)<=122:
	                return False
	        return True
	                
	        
	    def isupper(self, s):
	        for c in s:
	            if not 64<=ord(c)<=90:
	                return False
	        return True

### 运行结果

	Runtime: 20 ms, faster than 70.47% of Python online submissions for Detect Capital.
	Memory Usage: 13.6 MB, less than 8.81% of Python online submissions for Detect Capital.
	        
### 原题链接

https://leetcode.com/problems/detect-capital/



您的支持是我最大的动力
