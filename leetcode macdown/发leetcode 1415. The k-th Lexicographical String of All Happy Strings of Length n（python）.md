leetcode  1415. The k-th Lexicographical String of All Happy Strings of Length n（python）

### 描述

A happy string is a string that:

* consists only of letters of the set ['a', 'b', 'c'].
* s[i] != s[i + 1] for all values of i from 1 to s.length - 1 (string is 1-indexed).

For example, strings "abc", "ac", "b" and "abcbabcbcb" are all happy strings and strings "aa", "baa" and "ababbc" are not happy strings.

Given two integers n and k, consider a list of all happy strings of length n sorted in lexicographical order.

Return the kth string of this list or return an empty string if there are less than k happy strings of length n.



Example 1:


	Input: n = 1, k = 3
	Output: "c"
	Explanation: The list ["a", "b", "c"] contains all happy strings of length 1. The third string is "c".
	
Example 2:

	Input: n = 1, k = 4
	Output: ""
	Explanation: There are only 3 happy strings of length 1.


Example 3:


	Input: n = 3, k = 9
	Output: "cab"
	Explanation: There are 12 different happy string of length 3 ["aba", "abc", "aca", "acb", "bab", "bac", "bca", "bcb", "cab", "cac", "cba", "cbc"]. You will find the 9th string = "cab"
	
Example 4:


	Input: n = 2, k = 7
	Output: ""
	
Example 5:

	
	Input: n = 10, k = 100
	Output: "abacbabacb"

Note:

	1 <= n <= 10
	1 <= k <= 100



### 解析

根据题意，就是给出了一个数字 n 和一个数字 k ，要求使用 a 、b 、c 三个字符组成长度为 n 的字符串，而且要求每个字符串 s 满足 s[i] != s[i+1] ，最后将组成的所有的符合题意的字符串按照字典顺序进行排序，返回索引为 k-1 的元素，如果没有该元素则直接返回空字符串。思路比较简单：

* 判断 k 是否在题意所允许的大小范围内，如果超出范围，直接返回空字符串
* 初始化列表 l 为 ['a','b','c']
* 因为第一次已经初始化了 l ，所以只需要循环 n-1 次，在每次循环过程中，将之前的所有 l 中的元素都取出来，这些元素每一个都可以经过在末尾添加不同的其他两个字符形成新的俩个字符串，规律如下琐事，将这些形成的新字符串都添加到 l ，进行下一次的循环

		如最后一个字符为 'a' 的字符串 s ，可以形成 s+'b' 和 s+'c' 两个新的字符串
		如最后一个字符为 'b' 的字符串 s ，可以形成 s+'a' 和 s+'c' 两个新的字符串
		如最后一个字符为 'c' 的字符串 s ，可以形成 s+'a' 和 s+'a' 两个新的字符串
		

* 循环结束之后，得到的 l 中的元素已经是有顺序的，直接返回 l[k-1] 即可


### 解答
				

	class Solution(object):
	    def getHappyString(self, n, k):
	        """
	        :type n: int
	        :type k: int
	        :rtype: str
	        """
	        if k>3*(2**(n-1)):
	            return ""
	        l = ['a','b','c']
	        for _ in range(n-1):
	            N = len(l)
	            for s in range(N):
	                tmp = l.pop(0)
	                if tmp[-1] == 'a':
	                    for c in 'bc':
	                        l.append(tmp+c)
	                if tmp[-1] == 'b':
	                    for c in 'ac':
	                        l.append(tmp+c)
	                if tmp[-1] == 'c':
	                    for c in 'ab':
	                        l.append(tmp+c)
	        return l[k-1]
            	      
			
### 运行结果

	Runtime: 120 ms, faster than 35.00% of Python online submissions for The k-th Lexicographical String of All Happy Strings of Length n.
	Memory Usage: 13.3 MB, less than 87.50% of Python online submissions for The k-th Lexicographical String of All Happy Strings of Length n.
### 解析

将上面的过程可以改成递归形式

### 解答

	class Solution(object):
	    def getHappyString(self, n, k):
	        def r(n, path, happies):
	                if len(path) == n:
	                    happies.append(path)
	                    return
	                for x in ['a', 'b', 'c']:
	                    if not path or path[-1] != x:
	                        r(n, path + x, happies)
	                        
	        happies = []
	        r(n, "", happies)
	        if k > len(happies):
	            return ""
	        return happies[k - 1]
	        
### 运行结果

	Runtime: 116 ms, faster than 37.50% of Python online submissions for The k-th Lexicographical String of All Happy Strings of Length n.
	Memory Usage: 13.6 MB, less than 30.00% of Python online submissions for The k-th Lexicographical String of All Happy Strings of Length n.
	

原题链接：https://leetcode.com/problems/the-k-th-lexicographical-string-of-all-happy-strings-of-length-n/



您的支持是我最大的动力
