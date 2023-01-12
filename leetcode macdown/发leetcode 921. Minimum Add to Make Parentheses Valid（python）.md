leetcode 921. Minimum Add to Make Parentheses Valid （python）

### 描述


A parentheses string is valid if and only if:

* It is the empty string,
* It can be written as AB (A concatenated with B), where A and B are valid strings, or
* It can be written as (A), where A is a valid string.

You are given a parentheses string s. In one move, you can insert a parenthesis at any position of the string.

* For example, if s = "()))", you can insert an opening parenthesis to be "(()))" or a closing parenthesis to be "())))".


Return the minimum number of moves required to make s valid.


Example 1:


	Input: s = "())"
	Output: 1
	
Example 2:

	Input: s = "((("
	Output: 3


Example 3:


	Input: s = "()"
	Output: 0
	
Example 4:

	Input: s = "()))(("
	Output: 4

	


Note:

	1 <= s.length <= 1000
	s[i] is either '(' or ')'.



### 解析

根据题意，就是给出了一个左右圆括号组成的字符串，让我们判断这个字符串是否是合法的，并且在不合法的时候至少通过操作变成合法的需要次操作。

合法的括号字符串就是满足 ()() 或者 (()) 或者为空的时候。

我们需要进行的操作就是在任意的位置能插入任意的左或者右圆括号。

其实这个题思路比较简单，考察的就是栈的使用，只要找出不合法的圆括号有几个，那就至少需要进行多少次操作。因为只有 () 才是最小的合法括号，所以：

* 初始化 l 和 r 用来分别记录不合法的左右括号的出现的个数
* 从最后开始弹出元素，判断如果是 ) 则 r 加一
* 否则如果是 ( 且 r>0 ，那说明有最小单元的合法括号组合，r 减一
* 否则如果是 ( 且 r==0，那说明还没有出现右括号，l 加一
* 最后统计出的不合法的左右做好的个数之和即位答案

### 解答
				
	class Solution(object):
	    def minAddToMakeValid(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        if not s: return 0
	        if len(s) == 1: return 1
	        s = list(s)
	        r = 0
	        l = 0
	        while s:
	            parentheses = s.pop()
	            if parentheses == ')':
	                r += 1
	            elif parentheses == '(' and r > 0:
	                r -= 1
	            elif parentheses == '(' and r == 0:
	                l += 1
	        return l + r

            	      
			
### 运行结果


	Runtime: 16 ms, faster than 87.94% of Python online submissions for Minimum Add to Make Parentheses Valid.
	Memory Usage: 13.6 MB, less than 32.22% of Python online submissions for Minimum Add to Make Parentheses Valid.
	
### 解析

上面的解法理解起来比较麻烦，可以直接使用栈来存储遍历到的每个元素，只有当栈的最后一个元素为 ( 且当前的元素为 ) 才执行栈的弹出功能，否则将当前元素加入栈中，最后栈中存储的都是不合法的括号，即需要进行的最少的操作次数。

### 解答

	class Solution(object):
	    def minAddToMakeValid(self, s):
	        """
	        :type s: str
	        :rtype: int
	        """
	        if not s: return 0
	        if len(s) == 1: return 1
	        stack = [s[0]]
	        for c in s[1:]:
	            if stack and stack[-1]=='(' and c==')':
	                stack.pop()
	            else:
	                stack.append(c)
	        return len(stack)
	        
### 运行结果

	Runtime: 18 ms, faster than 61.95% of Python online submissions for Minimum Add to Make Parentheses Valid.
	Memory Usage: 13.5 MB, less than 32.22% of Python online submissions for Minimum Add to Make Parentheses Valid.

原题链接：https://leetcode.com/problems/minimum-add-to-make-parentheses-valid/



您的支持是我最大的动力
