leetcode  67. Add Binary（python）

### 描述

Given two binary strings a and b, return their sum as a binary string.




Example 1:

	Input: a = "11", b = "1"
	Output: "100"

	
Example 2:

	Input: a = "1010", b = "1011"
	Output: "10101"




Note:

	1 <= a.length, b.length <= 10^4
	a and b consist only of '0' or '1' characters.
	Each string does not contain leading zeros except for the zero itself.


### 解析


根据题意，给出了两个包含了 0 和 1 的二进制数字的字符串，题目要求我们将这两个字符串所表示的字面意思的二进制码进行相加，然后返回的二进制结果也用字符串表示。其实这个题就是考察加法的基本运算逻辑，其实是可以用 python 的内置函数进行快速求解的，但是思路比较简单，还是自己写代码吧：

* 初始化 M 和 N 得到 a 和 b 两个字符串的长度，保证 a 必须是较大的那个字符串
* 然后从它们的最后一位开始相加，满二往前进一
* 一直到遍历完 b 的所有字符，此时可能 a 还有需要遍历的字符，继续对 a 剩下的字符进行上面的相加操作
* 遍历结束所得到的字符串就是答案

### 解答
				
	class Solution(object):
	    def addBinary(self, a, b):
	        """
	        :type a: str
	        :type b: str
	        :rtype: str
	        """
	        M, N = len(a), len(b)
	        if M<N:
	            a,b = b,a
	            M,N = N,M
	        i,j = M-1,N-1
	        tmp = 0
	        result = ''
	        while i>-1 and j>-1:
	            c = int(a[i])+int(b[j])
	            if c + tmp > 1:
	                result = str(c+tmp-2) + result
	                tmp = 1
	            else:
	                result = str(c+tmp)+result
	                tmp = 0
	            i-=1
	            j-=1
	        while i>-1:
	            c = int(a[i]) + tmp
	            if c > 1:
	                result = str(c - 2) + result
	                tmp = 1
	            else:
	                result = str(c) + result
	                tmp = 0
	            i -= 1
	        if tmp>0:
	            result = '1' + result
	        return result

            	      
			
### 运行结果

	Runtime: 45 ms, faster than 10.19% of Python online submissions for Add Binary.
	Memory Usage: 13.5 MB, less than 50.08% of Python online submissions for Add Binary.

### 解析

上面的代码便于理解，但是太冗余了，简洁版本如下。

* 将 a 和 b 转换为列表 a 和 b ，初始化进位变量 tmp 为 0 ，结果 result 为空字符串
* 当 a 不为空或 b 不为空或 tmp 不为空的时候：

	（1）当 a 不为空 tmp += int(a.pop())
	
 	（2）当 b 不为空 tmp += int(b.pop())
 	
 	（3）然后将 tmp%2 得到的该位置数字拼接到 result 前面
 	
 	（4）tmp 需要整除 2
* 循环结束得到的 result 即为答案。

### 解答
	class Solution(object):
	    def addBinary(self, a, b):
	        """
	        :type a: str
	        :type b: str
	        :rtype: str
	        """
	        a = list(a)
	        b = list(b)
	        result = ''
	        tmp = 0
	        while a or b or tmp:
	            if a:
	                tmp += int(a.pop())
	            if b:
	                tmp += int(b.pop())
	            result = str(tmp%2) + result
	            tmp //= 2
	        return result
### 运行结果	        
	        
	Runtime: 29 ms, faster than 34.94% of Python online submissions for Add Binary.
	Memory Usage: 13.6 MB, less than 50.08% of Python online submissions for Add Binary.   


### 解析

虽然我不建议使用 python 的内置函数进行求解，但是当我用上面的方法通过之后，还是会手痒想使用内置函数解决一下，就算是复习一下之前的初级函数吧。主要就是用到了 int 的二进制相加和 bin 将十进制整数转换成二进制字符串结果。

### 解答

	class Solution(object):
	    def addBinary(self, a, b):
	        """
	        :type a: str
	        :type b: str
	        :rtype: str
	        """
	        return  bin(int(a,2)+int(b,2))[2:]	  
### 运行结果	        
	Runtime: 26 ms, faster than 48.42% of Python online submissions for Add Binary.
	Memory Usage: 13.6 MB, less than 22.38% of Python online submissions for Add Binary.        
	        
原题链接：https://leetcode.com/problems/add-binary/



您的支持是我最大的动力
