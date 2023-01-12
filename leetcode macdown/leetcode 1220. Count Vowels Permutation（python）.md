leetcode  1220. Count Vowels Permutation（python）




### 描述

Given an integer n, your task is to count how many strings of length n can be formed under the following rules:

* Each character is a lower case vowel ('a', 'e', 'i', 'o', 'u')
* Each vowel 'a' may only be followed by an 'e'.
* Each vowel 'e' may only be followed by an 'a' or an 'i'.
* Each vowel 'i' may not be followed by another 'i'.
* Each vowel 'o' may only be followed by an 'i' or a 'u'.
* Each vowel 'u' may only be followed by an 'a'.

Since the answer may be too large, return it modulo 10^9 + 7.





Example 1:

	Input: n = 1
	Output: 5
	Explanation: All possible strings are: "a", "e", "i" , "o" and "u".

	
Example 2:

	Input: n = 2
	Output: 10
	Explanation: All possible strings are: "ae", "ea", "ei", "ia", "ie", "io", "iu", "oi", "ou" and "ua".


Example 3:


	Input: n = 5
	Output: 68


Note:

	1 <= n <= 2 * 10^4



### 解析

根据题意，给定一个整数 n ，你的任务是计算在以下规则下可以形成多少个长度为 n 的字符串：

* 每个字符都是一个小写元音（'a', 'e', 'i', 'o', 'u'）
* 每个元音“a”后面只能跟一个“e”。
* 每个元音“e”后面只能跟一个“a”或“i”。
* 每个元音“i”后面不能跟着另一个“i”。
* 每个元音“o”后面只能跟一个“i”或“u”。
* 每个元音“u”后面只能跟一个“a”。

由于答案可能太大，因此以 10^9 + 7 为模返回。


* 我们将 a、e、i、o、u 分别用 0、1、2、3、4 表示，所以我们可以定义 dp[i][j] 表示长度为 i+1 的时候以字母 j 结尾的字符串数量，我们从后往前来写递推公式如下：

* dp[i][0] = dp[i + 1][1]
* dp[i][1] = (dp[i + 1][0] + dp[i + 1][2]) % MOD
* dp[i][2] = (dp[i + 1][0] + dp[i + 1][1] + dp[i + 1][3] + dp[i + 1][4]) % MOD
* dp[i][3] = (dp[i + 1][2] + dp[i + 1][4]) % MOD
* dp[i][4] = dp[i + 1][0]
​

计算完所有的 dp 值，将 dp 求和即可算出以不同字母结尾的长度为 n 的字符串数量，记得取模。

时间复杂度为 O(N\*5)，空间复杂度为 O(N\*5) 。


### 解答

	class Solution(object):
	    def countVowelPermutation(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        MOD = 10 ** 9 + 7
	        dp = [[0] * 5 for _ in range(n - 1)] + [[1] * 5]
	        for i in range(n-2, -1, -1):
	            dp[i][0] = dp[i + 1][1]
	            dp[i][1] = (dp[i + 1][0] + dp[i + 1][2]) % MOD
	            dp[i][2] = (dp[i + 1][0] + dp[i + 1][1] + dp[i + 1][3] + dp[i + 1][4]) % MOD
	            dp[i][3] = (dp[i + 1][2] + dp[i + 1][4]) % MOD
	            dp[i][4] = dp[i + 1][0]
	        return sum(dp[0][i] for i in range(5)) % MOD

### 运行结果

	Runtime: 269 ms, faster than 72.41% of Python online submissions for Count Vowels Permutation.
	Memory Usage: 17.6 MB, less than 27.59% of Python online submissions for Count Vowels Permutation.

### 解析

仔细观察可以发现，其实我们在已知上面谁在谁后面的情况下，可以知道谁在谁的前面：

* a 的前面只能是 e、i、u
* e 的前面只能是 a、i
* i 的前面只能是 e、o
* o 的前面只能是 i
* u 的前面只能是 i、o

我们将 a、e、i、o、u 分别用 0、1、2、3、4 表示，所以我们可以定义 dp[i][j] 表示长度为 i+1 的时候以字母 j 结尾的字符串数量，从前往后写递推公式如下：

* dp[i][0]=dp[i−1][1]+dp[i−1][2]+dp[i−1][4]
* dp[i][1]=dp[i−1][0]+dp[i−1][2]
* dp[i][2]=dp[i−1][1]+dp[i−1][3]
* dp[i][3]=dp[i−1][2]
* dp[i][4]=dp[i−1][2]+dp[i−1][3] 

这种的解法和上面的类似，并且还可以优化成一维的动态规划解法。时间复杂度为 O(N\*5)，空间复杂度为 O(5) 。


### 解答

	class Solution(object):
	    def countVowelPermutation(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        MOD = 10 ** 9 + 7
	        a, e, i, o, u = 1, 1, 1, 1, 1
	        for _ in range(n - 1):
	            a, e, i, o, u = ((e + i + u) % MOD, (a + i) % MOD, (e + o) % MOD, i, (i + o) % MOD)
	        return (a + e + i + o + u) % MOD
	



### 运行结果

	Runtime: 126 ms, faster than 93.10% of Python online submissions for Count Vowels Permutation.
	Memory Usage: 13.8 MB, less than 100.00% of Python online submissions for Count Vowels Permutation.

### 原题链接

https://leetcode.com/problems/count-vowels-permutation/


您的支持是我最大的动力
