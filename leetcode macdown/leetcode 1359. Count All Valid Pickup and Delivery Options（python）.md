leetcode 1359. Count All Valid Pickup and Delivery Options（python）




### 描述

Given n orders, each order consist in pickup and delivery services. 

Count all valid pickup/delivery possible sequences such that delivery(i) is always after of pickup(i). 

Since the answer may be too large, return it modulo 10^9 + 7.



Example 1:

	Input: n = 1
	Output: 1
	Explanation: Unique order (P1, D1), Delivery 1 always is after of Pickup 1.

	
Example 2:


	Input: n = 2
	Output: 6
	Explanation: All possible orders: 
	(P1,P2,D1,D2), (P1,P2,D2,D1), (P1,D1,P2,D2), (P2,P1,D1,D2), (P2,P1,D2,D1) and (P2,D2,P1,D1).
	This is an invalid order (P1,D2,P2,D1) because Pickup 2 is after of Delivery 2.

Example 3:

	Input: n = 3
	Output: 90

	



Note:

	1 <= n <= 500


### 解析


根据题意，给定 n 个订单，每个订单包含取货和送货服务。计算所有有效的取件/交付可能序列，以便 delivery(i)  始终在 pickup(i) 之后。由于答案可能太大，因此以 10^9 + 7 为模返回结果。

其实这道题我读完题目就知道这是在考察数学逻辑知识，因为这就是一个排列组合问题。我们已经限定死了某个快递件的取件动作一定在派送动作之前，那么：

1.n=1 的时候，当我们有一个快递件的时候，只能有一种送法 (P1, D1) 。

2.n=2 的时候，当我们有两个快递件的时候取件动作有两个 P1 和 P2 ，派送动作有两个 D1 和 D2：

* 假如我们在 n=1 的固定情况 (P1, D1) 下，将 P2 和 D2 看作整体插入，那么有三种方式：\_\_P1D1, P1\_\_D1, P1D1\_\_ ；
* 当 P2 和 D2 不挨着的时候，我们假定 P2 在 P1 之前的情况下，那么有两种方式：P2P1\_D1, P2P1D1\_ ；
* 当 P2 和 D2 不挨着的时候，我们假定 P2 在 P1 之后的情况下，那么有一种方式：P1P2D1\_ ，所以一共有  3+2+1 =6 种不同的送法。

3.n=3 的时候，我们已经有了 n=2 情况下的 6 种方式，我们选择任意一个 P1P2D1D2 来进行解释，我们在 P1P2D1D2 此基础上安排 P3 和 D3 ：

* 将 P3 和 D3 看作整体插入，那么有 5 种方法： \_\_P1P2D1D2, P1\_\_P2D1D2, P1P2\_\_D1D2, P1P2D1\_\_D2, P1P2D1\_\_D2 ；
* 然后当 P3 和 D3 不挨着的时候，我们假定 P3 在 P1 之前有 4 种方法：P3P1\_P2D1D2, P3P1P2\_D1D2, P3P1P2D1\_D2, P3P1P2D1D2\_ 
* 然后当 P3 和 D3 不挨着的时候，我们假定 P3 在 P2 之前有 3 种方法: P1P3P2\_D1D2, P1P3P2D1\_D2, P1P3P2D1D2\_
* 然后当 P3 和 D3 不挨着的时候，我们假定 P3 在 P2 之后有 2 种方法: P1P2P3D1\_D2, P1P2P3D1D2\_
* 然后当 P3 和 D3 不挨着的时候，我们假定 P3 在 D1 之后有 1 种方法: P1P2D1P3D2\_

所以当一共有 5+4+3+2+1=15 种，又因为 n=2 的时候有 6 种方法，所以，组合起来一共有 15x6=90 种方法

4.n=4 的时候，按照上面的逻辑进行推演，应该有 90x(7+6+5+4+3+2+1)=2520 种方法。

以此类推我们就知道了规律，当有 n (>=2) 件快递的时候，已知 n-1 的组合已经算出来为 num ，我们要根据 n-1 的每种情况来安排新的组合，每种之前的组合可以形成的新的组合一共有 (1+2+3...+(2\*n-2)+1) ，化简之后为 (2\*n-1)\*n ，然后再和 num 相乘即可得到 n 件快递的运送方法。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。


### 解答
				

	class Solution(object):
	    def countOrders(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        result = 1
	        for i in range(2, n+1):
	            result *= (2*i-1)*i
	        return result % (10**9 + 7)
            	      
			
### 运行结果

	Runtime: 30 ms, faster than 29.63% of Python online submissions for Count All Valid Pickup and Delivery Options.
	Memory Usage: 13.5 MB, less than 46.30% of Python online submissions for Count All Valid Pickup and Delivery Options.


### 原题链接


https://leetcode.com/problems/count-all-valid-pickup-and-delivery-options/


您的支持是我最大的动力
