leetcode  1111. Maximum Nesting Depth of Two Valid Parentheses Strings（python）

### 描述


A string is a valid parentheses string (denoted VPS) if and only if it consists of "(" and ")" characters only, and:

* It is the empty string, or
* It can be written as AB (A concatenated with B), where A and B are VPS's, or
* It can be written as (A), where A is a VPS.

We can similarly define the nesting depth depth(S) of any VPS S as follows:

* depth("") = 0
* depth(A + B) = max(depth(A), depth(B)), where A and B are VPS's
* depth("(" + A + ")") = 1 + depth(A), where A is a VPS.

For example,  "", "()()", and "()(()())" are VPS's (with nesting depths 0, 1, and 2), and ")(" and "(()" are not VPS's.

 

Given a VPS seq, split it into two disjoint subsequences A and B, such that A and B are VPS's (and A.length + B.length = seq.length).

Now choose any such A and B such that max(depth(A), depth(B)) is the minimum possible value.

Return an answer array (of length seq.length) that encodes such a choice of A and B:  answer[i] = 0 if seq[i] is part of A, else answer[i] = 1.  Note that even though multiple answers may exist, you may return any of them.


Example 1:

	Input: seq = "(()())"
	Output: [0,1,1,1,1,0]

	
Example 2:


	Input: seq = "()(())()"
	Output: [0,0,0,1,1,0,1,1]


Note:


	1 <= seq.size <= 10000

### 解析


根据题意，给出了一个字符串 seq 只包含了 "(" 和 ")" 两种符号，这种合法的括号字符串被称为 VPS，并且：

* 它可以是一个空字符串，或者
* 它也可以记作 AB（A 与 B 拼接），其中 A 和 B 都是 VPS ，或者
* 它也可以记作 (A)，其中 A 是有 VPS 。

同时题目中给出了 VPS 的深度计算方法定义为 depth(S)：

* depth("") = 0
* depth(A + B) = max(depth(A), depth(B))，其中 A 和 B 都是 VPS
* depth("(" + A + ")") = 1 + depth(A)，其中 A 是 VPS

例如，""、"()()" 和 "()(()())" 都是 VPS（嵌套深度分别为 0、1、2），而 ")(" 和 "(()" 都不是 VPS 。

给你一个 VPS 变量 seq，将其分成两个不相交的子序列 A 和 B，且 A 和 B 满足 VPS 的定义（A.length + B.length = seq.length）。

拆分的结果有很多中，题目要求从中选出任意一组 A 和 B，使 max(depth(A), depth(B)) 的可能取值最小。

返回长度为 seq.length 答案数组 answer ，同时对分解出来的 A 和 B 分别进行编码，规则是：如果 seq[i] 是 A 的一部分，那么 answer[i] = 0。否则，answer[i] = 1。题目中有多个满足要求的答案存在，只需返回其中一个。

其实题目虽然很绕，但是理解之后也比较简单，要做的只有两件事：

* 第一件事就是找出 seq 的最深深度 d
* 第二件事就是将 seq 分成两个子序列 A 和 B ，保证使 max(depth(A), depth(B)) 的可能取值最小。

最简单的方法就是进行两次遍历：

* 第一次遍历就是找出最深的深度 d ，并且使用列表 record 记录每个位置的深度
* 第二次遍历就是将小于等于 d/2 的深度的字符都分给 A ，其他的字符分给 B ，最后将形成的结果列表 result 返回即可。因为假如当最深深度是 4 ，只有将其分为两个深度为 2 的 A 和 B 才能保证 max(depth(A), depth(B)) 的可能取值最小，假如当最深深度为 5 ，只有将其分为两个深度为 2 和 3 的 A 和 B 才能保证 max(depth(A), depth(B)) 的可能取值最小


### 解答
				
	class Solution(object):
	    def maxDepthAfterSplit(self, seq):
	        """
	        :type seq: str
	        :rtype: List[int]
	        """
	        if len(seq)<3 : return [0]*len(seq)
	        max_d = 0
	        cur_d = 0
	        record = []
	        stack = []
	        result = []
	        for c in seq:
	            if c=='(':
	                cur_d += 1
	                record.append(cur_d)
	            else:
	                record.append(cur_d)
	                cur_d -= 1
	            max_d = max(max_d, cur_d)
	        half = max_d/2
	        for i in record:
	            if i<= half:
	                result.append(0)
	            else:
	                result.append(1)
	        return result
	                
	                
	        

            	      
			
### 运行结果
		
	Runtime: 36 ms, faster than 87.50% of Python online submissions for Maximum Nesting Depth of Two Valid Parentheses Strings.
	Memory Usage: 13.7 MB, less than 100.00% of Python online submissions for Maximum Nesting Depth of Two Valid Parentheses Strings.


原题链接：https://leetcode.com/problems/maximum-nesting-depth-of-two-valid-parentheses-strings/



您的支持是我最大的动力
