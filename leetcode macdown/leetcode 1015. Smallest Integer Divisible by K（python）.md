leetcode  1015. Smallest Integer Divisible by K（python）

### 每日经典

《永遇乐·京口北固亭怀古》 ——辛弃疾（宋）

千古江山，英雄无觅、孙仲谋处。舞榭歌台，风流总被、雨打风吹去。斜阳草树，寻常巷陌，人道寄奴曾住。想当年，金戈铁马，气吞万里如虎。

元嘉草草，封狼居胥，赢得仓皇北顾。四十三年，望中犹记，烽火扬州路。可堪回首，佛狸祠下，一片神鸦社鼓。凭谁问：廉颇老矣，尚能饭否？


### 描述

Given a positive integer k, you need to find the length of the smallest positive integer n such that n is divisible by k, and n only contains the digit 1.

Return the length of n. If there is no such n, return -1.

Note: n may not fit in a 64-bit signed integer.



Example 1:


	Input: k = 1
	Output: 1
	Explanation: The smallest answer is n = 1, which has length 1.
	
Example 2:

	
	Input: k = 2
	Output: -1
	Explanation: There is no such positive integer n divisible by 2.

Example 3:

	Input: k = 3
	Output: 3
	Explanation: The smallest answer is n = 111, which has length 3.

	


Note:

	1 <= k <= 10^5



### 解析

根据题意，给定一个正整数 k，需要找到最小正整数 n ，使得 n 可以被 k 整除，并且 n 只包含数字 1 ，返回 n 的长度。 如果没有这样的 n ，则返回 -1 。

读完题之后其实我们发现这个是一道找规律的题目，我们随便取 k = 3 ，列举几个例子

	1%3 = 1
	1*10+1=11%3=2      
	11*10+1=111%3=0
	111*10+1=1111%3=1
	1111*10+1=11111%3=2
	11111*10+1=111111%3=0
	
我们发现于对 k=3 取模是有规律的，如 11 对 3 取模，就是 1 对 3 取模的结果乘 10 然后加 1 的值对 3 取模的结果，以此类推，111 对 3 取模，就是 11 对 3 取模的结果乘 10 然后加 1 的值对 3 取模的结果，通过这个规律，这样我们就能算出来 n 位数的 1 组成的正整数对 k 取模的规律，这些取模的结果存到一个集合里面，如果出现取模的结果为 0 ，说明可以整除，直接返回 n ；如果取模的结果已经在集合中出现，说明之后就算再多位的 1 组成的正整数对 k 取模都会重复，没有结果，直接返回 0 。

另外看大神在最开始加入下面的代码说可以提速，但是没搞懂是为什么，而且实际的运行效果发现好像没有提速，耗时反而多了 8 ms ：

	if k % 10 not in {1, 3, 7, 9}: return -1


### 解答
				
	
	class Solution(object):
	    def smallestRepunitDivByK(self, k):
	        """
	        :type k: int
	        :rtype: int
	        """
	        mod = 0
	        mods = set()
	        for n in range(1, k+1):
	            mod = (mod * 10 + 1) % k
	            if mod==0:
	                return n
	            if mod in mods:
	                return -1
	            mods.add(mod)
	        return -1
            	      
			
### 运行结果

	Runtime: 48 ms, faster than 71.43% of Python online submissions for Smallest Integer Divisible by K.
	Memory Usage: 18 MB, less than 14.29% of Python online submissions for Smallest Integer Divisible by K.


原题链接：https://leetcode.com/problems/smallest-integer-divisible-by-k/



您的支持是我最大的动力
