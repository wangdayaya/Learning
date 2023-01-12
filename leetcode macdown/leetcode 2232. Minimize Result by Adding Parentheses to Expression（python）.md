leetcode  2232. Minimize Result by Adding Parentheses to Expression（python）




### 描述


You are given a 0-indexed string expression of the form "<num1>+<num2>" where <num1> and <num2> represent positive integers.

Add a pair of parentheses to expression such that after the addition of parentheses, expression is a valid mathematical expression and evaluates to the smallest possible value. The left parenthesis must be added to the left of '+' and the right parenthesis must be added to the right of '+'.

Return expression after adding a pair of parentheses such that expression evaluates to the smallest possible value. If there are multiple answers that yield the same result, return any of them.

The input has been generated such that the original value of expression, and the value of expression after adding any pair of parentheses that meets the requirements fits within a signed 32-bit integer.


Example 1:

	Input: expression = "247+38"
	Output: "2(47+38)"
	Explanation: The expression evaluates to 2 * (47 + 38) = 2 * 85 = 170.
	Note that "2(4)7+38" is invalid because the right parenthesis must be to the right of the '+'.
	It can be shown that 170 is the smallest possible value.

	
Example 2:

	Input: expression = "12+34"
	Output: "1(2+3)4"
	Explanation: The expression evaluates to 1 * (2 + 3) * 4 = 1 * 5 * 4 = 20.


Example 3:

	Input: expression = "999+999"
	Output: "(999+999)"
	Explanation: The expression evaluates to 999 + 999 = 1998.





Note:

	3 <= expression.length <= 10
	expression consists of digits from '1' to '9' and '+'.
	expression starts and ends with digits.
	expression contains exactly one '+'.
	The original value of expression, and the value of expression after adding any pair of parentheses that meets the requirements fits within a signed 32-bit integer.


### 解析

根据题意，给定一个格式为“<num1>+<num2>”的 0 索引字符串表达式，其中 <num1> 和 <num2> 表示正整数。

向表达式添加一对括号之后，表达式是一个有效的数学表达式，其实就是括号内的进行加法运算，括号外的进行乘法运算，要求我们计算可能出现的最小结果值所对应的字符串，有不懂的结合看一下给出的几个例子就明白了。

因为在不同的地方都可以加一堆括号，所以如果有多个答案则返回其中任何一个。

这道题考察的其实就是对题目的理解和暴力求解，我在比赛的时候就是用暴力解决的，因为限制条件中 expression 的最大长度为 10 ，而且在比赛过程中暴力解决方法可以省去很多思考的时间，同样能 AC 。但是比赛之后我去看论坛想寻求更加简单的解法，发现都是用的暴力解法【捂脸】，同样又失去了优化代码的动力。

我的思路其实很简单：

* 将 expression 中的两个数字提取出来分别为 A 或者 B 
* 使用两个变量 i 和 j 表示左括号在 A 的位置和右括号在 B 的位置，进行两重循环遍历
* 因为“(” 可以把 A 切割成两部分 preA 和 tailA （当然如果 preA 为空，因为他的位置要进行乘法所以要给他赋值为 1 ；如果 tailA 为空，因为他的位置要进行加法所以要给他赋值为 0 ）；“)”可以把 B 切割成两部分 preB 和 tailB （对于为空的情况，和上面的处理方法一样）
* 计算 preA\*(tailA + preB)\*tailB ，如果小于当前的 mx ，则更新最后的结果字符串 result 
* 循环结束直接返回 result 即可

N1 和 N2 分别为两个数字的字符串长度，时间复杂度为 O(N1 \* N2) ，空间复杂度为 O(N1 + N2)。

### 解答
				

	class Solution(object):
	    def minimizeResult(self, expression):
	        """
	        :type expression: str
	        :rtype: str
	        """
	        A,B = expression.split('+')
	        n1 = len(A)
	        n2 = len(B)
	        mx = float('inf')
	        result = ''
	        for i in range(n1):
	            preA = A[:i]
	            if not preA:
	                preA = 1
	            tailA = A[i:]
	            if not tailA:
	                tailA = 0
	            for j in range(1, n2+1):
	                preB = B[:j]
	                if not preB:
	                    preB = 0
	                tailB = B[j:]
	                if not tailB:
	                    tailB = 1
	                tmp = int(preA)*(int(tailA) + int(preB))*int(tailB)
	                if mx > tmp:
	                    mx = min(mx, tmp)
	                    result  = A[:i] + '(' + A[i:] + '+' + B[:j] + ')' + B[j:]
	        return result
            	      
			
### 运行结果


	124 / 124 test cases passed.
	Status: Accepted
	Runtime: 34 ms
	Memory Usage: 13.6 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-288/problems/minimize-result-by-adding-parentheses-to-expression/


您的支持是我最大的动力
