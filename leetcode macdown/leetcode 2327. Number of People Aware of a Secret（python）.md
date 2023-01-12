leetcode 2327. Number of People Aware of a Secret （python）



### 描述


On day 1, one person discovers a secret.

You are given an integer delay, which means that each person will share the secret with a new person every day, starting from delay days after discovering the secret. You are also given an integer forget, which means that each person will forget the secret forget days after discovering it. A person cannot share the secret on the same day they forgot it, or on any day afterwards.

Given an integer n, return the number of people who know the secret at the end of day n. Since the answer may be very large, return it modulo 10^9 + 7.


Example 1:


	Input: n = 6, delay = 2, forget = 4
	Output: 5
	Explanation:
	Day 1: Suppose the first person is named A. (1 person)
	Day 2: A is the only person who knows the secret. (1 person)
	Day 3: A shares the secret with a new person, B. (2 people)
	Day 4: A shares the secret with a new person, C. (3 people)
	Day 5: A forgets the secret, and B shares the secret with a new person, D. (3 people)
	Day 6: B shares the secret with E, and C shares the secret with F. (5 people)
	
Example 2:

	Input: n = 4, delay = 1, forget = 3
	Output: 6
	Explanation:
	Day 1: The first person is named A. (1 person)
	Day 2: A shares the secret with B. (2 people)
	Day 3: A and B share the secret with 2 new people, C and D. (4 people)
	Day 4: A forgets the secret. B, C, and D share the secret with 3 new people. (6 people)






Note:

	2 <= n <= 1000
	1 <= delay < forget <= n


### 解析

根据题意，第一天一个人发现了一个秘密。给定一个整数 delay ，这意味着每个人从发现秘密后 delay 天后开始每天都会与一个新人分享这个秘密，给定一个整数 forget ，这意味着每个人都会在发现秘密后的 forget 天后忘记它。 一个人不能在他们忘记秘密的当天或之后的任何一天分享秘密。给定一个整数 n ，返回在第 n 天结束时知道秘密的人数。 由于答案可能非常大，因此以 10^9 + 7 为模返回。

这种题我们可以使用动态规划来解答，因为在新的一天我们既要算知道秘密的新人，又要算忘记秘密的人，如果硬抠细节这很绕，所以我们干脆使用 dp 来保存第 i 天新增的指导秘密的人，第 i 天知道秘密的人又会对 [i+delay, i+ forget) 区间有贡献，所以最后一共知道秘密的人，就是从最后往前推 forget 天以内的新增人数的总和。

时间复杂度为 O(N)  ，空间复杂度为 O(N) 。



### 解答
				
	class Solution(object):
	    def peopleAwareOfSecret(self, n, delay, forget):
	        """
	        :type n: int
	        :type delay: int
	        :type forget: int
	        :rtype: int
	        """
	        dp = [0] * (n+1)
	        dp[1] = 1
	        result = 0
	        for i in range(1, n+1):
	            for j in range(i+delay, i+forget):
	                if j<n+1:
	                    dp[j] += dp[i]
	        for i in range(n-forget+1, n+1):
	            result += dp[i]
	        return result % (10**9+7)

            	      
			
### 运行结果

	
	82 / 82 test cases passed.
	Status: Accepted
	Runtime: 536 ms
	Memory Usage: 13.5 MB
	Submitted: 0 minutes ago



### 原题链接

https://leetcode.com/contest/weekly-contest-300/problems/number-of-people-aware-of-a-secret/

您的支持是我最大的动力
