leetcode 344. Reverse String （python）




### 描述



Write a function that reverses a string. The input string is given as an array of characters s.

You must do this by modifying the input array in-place with O(1) extra memory.



Example 1:

	Input: s = ["h","e","l","l","o"]
	Output: ["o","l","l","e","h"]

	
Example 2:


	Input: s = ["H","a","n","n","a","h"]
	Output: ["h","a","n","n","a","H"]





Note:

	1 <= s.length <= 10^5
	s[i] is a printable ascii character.


### 解析

根据题意，编写一个反转字符串的函数。 输入的字符串参数以字符数组 s 的形式给出。题目要求我们使用 O(1) 额外内存修改输入的数组来做到这一点。

大家知道字符串其实是不可变对象，所以这里的将输入的字符串以字符数组 s 的形式给出来了，其实这也就是暗示大家用数组的思想来解题，这样就能做到不开辟额外的内存空间。

而且这里不要想投机取巧，直接用一些函数将 s 用额外的空间逆序排序返回，这是没有用的， 不信你可以写代码试试，题目的审题机制只会去查看 s 的变化情况。

其实这道题就是考察一个对数组元素进行交换的基本操作，我们已经有了数组 s ，所以我们定义一个索引 i 和一个索引 j ，然后当 i < j 的时候进行 while 循环，在循环的里头
交换 s[j] 和 s[i] 的值即可，然后 i 加一，j 减一。当 while 循环结束之后，返回 s 即可。其实这里不返回 s 也是可以的，因为刚才说过了，题目审题机制会自动去检查 s ，不信你可以写代码试试。

这道题的难度只有 Eazy ，所以还是很简单的，时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答
				

	class Solution(object):
	    def reverseString(self, s):
	        i = 0
	        j = len(s) - 1
	        while i<j:
	            s[i],s[j] = s[j],s[i]
	            i += 1
	            j -= 1
	        return s
	        
            	      
			
### 运行结果

	Runtime: 172 ms, faster than 84.45% of Python online submissions for Reverse String.
	Memory Usage: 20.9 MB, less than 98.55% of Python online submissions for Reverse String.



### 原题链接


https://leetcode.com/problems/reverse-string/


您的支持是我最大的动力
