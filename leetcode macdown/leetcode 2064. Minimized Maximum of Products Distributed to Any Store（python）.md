leetcode  2064. Minimized Maximum of Products Distributed to Any Store（python）

### 描述


You are given an integer n indicating there are n specialty retail stores. There are m product types of varying amounts, which are given as a 0-indexed integer array quantities, where quantities[i] represents the number of products of the i<sup>th</sup> product type.

You need to distribute all products to the retail stores following these rules:

* A store can only be given at most one product type but can be given any amount of it.
* After distribution, each store will have been given some number of products (possibly 0). Let x represent the maximum number of products given to any store. You want x to be as small as possible, i.e., you want to minimize the maximum number of products that are given to any store.

Return the minimum possible x.


Example 1:

	Input: n = 6, quantities = [11,6]
	Output: 3
	Explanation: One optimal way is:
	- The 11 products of type 0 are distributed to the first four stores in these amounts: 2, 3, 3, 3
	- The 6 products of type 1 are distributed to the other two stores in these amounts: 3, 3
	The maximum number of products given to any store is max(2, 3, 3, 3, 3, 3) = 3.

	
Example 2:

	Input: n = 7, quantities = [15,10,10]
	Output: 5
	Explanation: One optimal way is:
	- The 15 products of type 0 are distributed to the first three stores in these amounts: 5, 5, 5
	- The 10 products of type 1 are distributed to the next two stores in these amounts: 5, 5
	- The 10 products of type 2 are distributed to the last two stores in these amounts: 5, 5
	The maximum number of products given to any store is max(5, 5, 5, 5, 5, 5, 5) = 5.


Example 3:

	Input: n = 1, quantities = [100000]
	Output: 100000
	Explanation: The only optimal way is:
	- The 100000 products of type 0 are distributed to the only store.
	The maximum number of products given to any store is max(100000) = 100000.

	




Note:

	m == quantities.length
	1 <= m <= n <= 10^5
	1 <= quantities[i] <= 10^5


### 解析

根据题意，给定一个整数 n ，表示有 n 个专卖店。 有 m 种不同数量的产品类型，它们以 0 索引整数数组 quantities 给出，其中 quantity[i] 表示第 i 种产品类型的产品数量。题目需要我们按照以下规则将所有产品分发到专卖店：

* 一个商店最多只能展示一种产品类型，但不限数量
* 分发后，每家商店都会获得一定数量的各类产品（可能为 0 ）。 设 x 代表任何商店展示的产品总量， 题意要求 x 尽可能小。

题目要求最后返回可能的最小的最大值 x 。假设每个商店最多有 limit 件产品，如果分给每个商品的商店越多，那么 limit 越小，如果分给每个商品的商店越少，那么 limit 越大，但是总的商店数量是有限的 n ，这就会报错，这么一分析其实就是二分搜值法。




### 解答
				

	class Solution(object):
	    def minimizedMaximum(self, n, quantities):
	        """
	        :type n: int
	        :type quantities: List[int]
	        :rtype: int
	        """
	        low = 1
	        high = max(quantities)
	        while low < high:
	            limit = low + (high - low) // 2
	            if self.checkOk(quantities, n, limit):
	                high = limit
	            else:
	                low = limit + 1
	        return low
	    
	    def checkOk(self, quantities, n, limit):
	        result = 0
	        for q in quantities:
	            if q % limit == 0:
	                result += q // limit
	            else:
	                result += q // limit + 1
	        return result <= n
			
### 运行结果

	
	Runtime: 2128 ms, faster than 58.62% of Python online submissions for Minimized Maximum of Products Distributed to Any Store.
	Memory Usage: 24.3 MB, less than 50.00% of Python online submissions for Minimized Maximum of Products Distributed to Any Store.

原题链接：https://leetcode.com/problems/minimized-maximum-of-products-distributed-to-any-store/



您的支持是我最大的动力
