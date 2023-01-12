leetcode  20. Valid Parentheses（python）

### 描述


Given a string s containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:

* Open brackets must be closed by the same type of brackets.
* Open brackets must be closed in the correct order.



Example 1:


	Input: s = "()"
	Output: true
	
Example 2:

	Input: s = "()[]{}"
	Output: true


Example 3:

	Input: s = "(]"
	Output: false

	
Example 4:

	Input: s = "([)]"
	Output: false

	
Example 5:


	Input: s = "{[]}"
	Output: true

Note:

	1 <= s.length <= 10^4
	s consists of parentheses only '()[]{}'.


### 解析

根据题意，就是判断给出的三种括号符号所组成的字符串是否适合法的，提中规定合法的括号字符串必须满足两点：

* 左括号必须用同类型的右括号相对应
* 左括号必须以正确的顺序关闭

如果还不懂的话那就是结合例子可以知道，无非就是有左括号就必须要有相同的右括号，如“()[]{}”；而且不同的括号之间不能非法嵌套，如：“([)]”；而是以完整的左右括号形式嵌套在另一组括号中。如“{[]}”。

说到这里解题思路就出来了，因为最内层嵌套的括号肯定是一组完整的左右类型相同的一组括号，如““{[][]{()}()}””，所以要判断这个括号字符串是否合法，只需要不断将 () 或者 [] 或者 {} 替换成空字符串就可以，如果在执行了替换操作之后字符串长度没有发生变化，那么可以直接返回 False 表示该括号字符串 s 是非法的，否则一直循环替换操作直到 s 成为空字符串再返回 True ，表示该符号字符串是合法的。结合代码看更容易理解。


### 解答
					
	class Solution(object):
	    def isValid(self, s):
	        """
	        :type s: str
	        :rtype: bool
	        """
	        while len(s) > 0:
	            t = len(s)
	            s = s.replace('()', '')
	            s = s.replace('{}', '')
	            s = s.replace('[]', '')
	            if t == len(s): return False
	        return True

            	      
			
### 运行结果
	Runtime: 48 ms, faster than 7.62% of Python online submissions for Valid Parentheses.
	Memory Usage: 13.9 MB, less than 12.31% of Python online submissions for Valid Parentheses.


### 解析

另外我们可以用栈的思想来解决这个题，因为括号是必须是有所对应的，所以我们可以遍历括号所组成的字符串 s ，初始化一个栈 stack ，然后将第一个元素放入 stack 中，从第二个元素开始遍历，如果该元素是 stack 中最后一个元素所对应的括号，那么执行 stack.pop() 将最后一个元素去掉，否则将该元素追加到 stack 中，一直按照这种方式遍历结束，如果最后的结果是空字符串则返回 True 表示 s 是合法的，否则返回 False 表示 s 是不合法的。结合代码更容易理解，当然还可以进一步简化，代码过程，自己可以思考优化一下。

### 解答	

	class Solution(object):
	    def isValid(self, s):
	        """
	        :type s: str
	        :rtype: bool
	        """
	        stack = [s[0]]
	        for c in s[1:]:
	            if c == '(' or c == '{' or c == '[':
	                stack.append(c)
	            elif not stack:
	                return False
	            elif c == ')' and '(' != stack.pop():
	                return False
	            elif c == '}' and '{' != stack.pop():
	                return False
	            elif c == ']' and '[' != stack.pop():
	                return False
	        return len(stack)==0            	      
			
### 运行结果

	Runtime: 24 ms, faster than 39.53% of Python online submissions for Valid Parentheses.
	Memory Usage: 13.6 MB, less than 59.80% of Python online submissions for Valid Parentheses.
	


原题链接：https://leetcode.com/problems/valid-parentheses/



您的支持是我最大的动力
