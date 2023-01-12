leetcode  1641. Count Sorted Vowel Strings（python）

### 描述


Given an integer n, return the number of strings of length n that consist only of vowels (a, e, i, o, u) and are lexicographically sorted.

A string s is lexicographically sorted if for all valid i, s[i] is the same as or comes before s[i+1] in the alphabet.




Example 1:

	Input: n = 1
	Output: 5
	Explanation: The 5 sorted strings that consist of vowels only are ["a","e","i","o","u"].	
	
Example 2:

	Input: n = 2
	Output: 15
	Explanation: The 15 sorted strings that consist of vowels only are
	["aa","ae","ai","ao","au","ee","ei","eo","eu","ii","io","iu","oo","ou","uu"].
	Note that "ea" is not a valid string since 'e' comes after 'a' in the alphabet.

Example 3:

	Input: n = 33
	Output: 66045



Note:

	1 <= n <= 50 

### 解析

根据题意，只要先找出规律就可以用动态规划的思想去做，首先找规律，当 n 为 1 的时候，以每种元音开头的合法字符串有多少种，可以这样如下列出：

		a e i o u
		1 1 1 1 1
		
当 n 为 2 的时候，可以如下列出：

		a e i o u
		1 1 1 1 1
		5 4 3 2 1
	
当 n 为 3 的时候，可以如下列出：

		a e i o u
		1  1  1  1  1
		5  4  3  2  1
		15 10 6  3  1
		
以此类推，其中的规律很明显，初始化一个数组 res ，res 有 5 个元素，没个元素对应的是分别以 5 个元音开头的字符串可以出现的合法次数，res[1] 表示的是以 a 开头的合法字符粗汉，以此类推。遍历 n 次，每次遍历的时候，只需要将 res[j] 变为 sum(res[j:]) 即可，遍历结束求 sum(res) 即可得到答案。

### 解答
				
	class Solution(object):
	    def countVowelStrings(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
        	 res = [1,1,1,1,1]
		    for i in range(1, n):
		        for j in range(5):
		            res[j] = sum(res[j:])
		    return sum(res)
            	      
			
### 运行结果

	Runtime: 16 ms, faster than 86.61% of Python online submissions for Count Sorted Vowel Strings.
	Memory Usage: 13.3 MB, less than 93.61% of Python online submissions for Count Sorted Vowel Strings.

### 解析

也可以使用 DFS 来解决， n 就相当于 DFS 的递归深度，宽度就是 5 ，因为是 5 个元音字母之间去排列组合，但是因为递归栈的深度太深，导致耗时严重，刚刚可以 AC ，这时就更显出上面动态规划的省时省力了。

### 解答

	class Solution(object):
	    def __init__(self):
	        self.result = 0
	    
	    def countVowelStrings(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        def dfs(idx, length, L, vowels):
	            if length == n:
	                self.result += 1
	                return
	            for i in range(idx, len(vowels)):
	                dfs(i, length + 1, L + vowels[i], vowels)
	
	        dfs(0, 0, '', 'aeiou')
	        return self.result
### 运行结果

	Runtime: 7964 ms, faster than 5.56% of Python online submissions for Count Sorted Vowel Strings.
	Memory Usage: 13.3 MB, less than 83.33% of Python online submissions for Count Sorted Vowel Strings.

原题链接：https://leetcode.com/problems/count-sorted-vowel-strings

