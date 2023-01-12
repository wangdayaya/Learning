leetcode  2320. Count Number of Ways to Place Houses（python）




### 描述

There is a street with n * 2 plots, where there are n plots on each side of the street. The plots on each side are numbered from 1 to n. On each plot, a house can be placed. Return the number of ways houses can be placed such that no two houses are adjacent to each other on the same side of the street. Since the answer may be very large, return it modulo 10^9 + 7.

Note that if a house is placed on the ith plot on one side of the street, a house can also be placed on the ith plot on the other side of the street.



Example 1:

	Input: n = 1
	Output: 4
	Explanation: 
	Possible arrangements:
	1. All plots are empty.
	2. A house is placed on one side of the street.
	3. A house is placed on the other side of the street.
	4. Two houses are placed, one on each side of the street.

	
Example 2:

![](https://assets.leetcode.com/uploads/2022/05/12/arrangements.png)

	Input: n = 2
	Output: 9
	Explanation: The 9 possible arrangements are shown in the diagram above.





Note:

* 1 <= n <= 10^4


### 暴力 DFS

根据题意，有一条街道有 n * 2 个地块，街道的每一侧都有 n 个地块。 每边的地块从 1 到 n 编号。 在每个地块上，可以放置一所房子。返回可以放置房屋的方式数，以使街道的同一侧没有两个房屋彼此相邻。 由于答案可能非常大，因此以 10^9 + 7 为模返回。

请注意，如果房子放在街道一侧的第 i 个地块上，那么房子也可以放在街道另一侧的第 i 个地块上。

其实通过找规律我们可以发现，：

	n = 1 时候，一侧放置房屋方式数为 2 ，两侧总共可以放置房屋方式数为 4
	n = 2 时候，一侧放置房屋方式数为 3 ，两侧总共可以放置房屋方式数为 9
	n = 3 时候，一侧放置房屋方式数为 5 ，两侧总共可以放置房屋方式数为 25
	n = 4 时候，一侧放置房屋方式数为 8 ，两侧总共可以放置房屋方式数为 64
	...

可以发现随着 n 的增加，一侧放置房屋方式数是斐波那契数列，我们只要知道 n 时一侧放置房屋方式数，然后计算平方再取  10^9 + 7 的模即可。

所以我们的关键是解决斐波那契数列的计算，这里我们先使用最无脑的暴力 DFS 进行解题递归规律就是 :

	dfs(n) = dfs(n-1)+dfs(n-2)
根据但是我预测会超时（其实我就是运行报错了），因为上面限制条件中写了 n 最大是 10000 ，我们在进行递归的时候是自顶向下，会有很多重复的计算，如计算 dfs(10) 就要先进行 dfs(9) 和 dfs(8) 的计算，但是在计算  dfs(9) 的时候又计算了一次 dfs(8) ，以此类推，所以在 n 为 10000 的时候肯定会超时，dfs 计算过程就像是一棵二叉树自顶向下分裂。 

时间复杂度为 O(2^N)， 空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def countHousePlacements(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        if n == 1: return 4
	        if n == 2: return 9
	
	        def dfs(n):
	            if n == 1: return 2
	            if n == 2: return 3
	            return dfs(n - 2) + dfs(n - 1)
	
	        return pow(dfs(n), 2) % (10 ** 9 + 7)

### 运行结果

	 Time Limit Exceeded 
### 记忆化 DFS

果然不出所料超时了，其实我们已经分析了原因了，无非就是有重复的计算，所以我们只要加入了记忆化，这样使用了记忆化的 DFS ，在计算 dfs(10) 就要先进行 dfs(9) 和 dfs(8) 的计算，但是在计算  dfs(9) 的时候我们需要的 dfs(8) 已经计算并保存下来了，所以我们只需要直接使用结果即可，这样相当于把一棵自顶向下的二叉树进行了剪枝操作，将重复计算的过程都去掉了，我们的整个计算过程都只是计算了 dfs(n) 、dfs(n-1) 、...、dfs(1) 一遍，没有多余的操作。

所以时间复杂度为 O(N) ，空间复杂度为 O (N) 。

### 解答

	class Solution(object):
	    def countHousePlacements(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        if n == 1: return 4
	        if n == 2: return 9
	        d = {1:2, 2:3}
	        def dfs(n):
	            if n == 1: return 2
	            if n == 2: return 3
	            if n in d:
	                return d[n]
	            d[n] = dfs(n - 2) + dfs(n - 1)
	            return d[n] 
	        return pow(dfs(n), 2) % (10 ** 9 + 7)

### 运行结果

	150 / 150 test cases passed.
	Status: Accepted
	Runtime: 757 ms
	Memory Usage: 61.6 MB

### 动态规划

其实细心的同学已经发现了上面的解法尽管使用了 DFS ，但是其实已经有了状态转移方程 :

	dfs(n) = dfs(n - 2) + dfs(n - 1)
	
所以我们可以使用动态规划来解题，动态规划和 DFS 不同之处在于，DFS 是自顶向下进行计算然后又把结果逐层往上返回到顶，而动态规划的计算时自底向上的，从 dp[1] 、dp[2] 开始，利用状态转移方程直接一次性计算到 dp[n] 。

时间复杂度为 O(N) ，空间复杂度为 O(N)。这里虽然时间复杂度和空间复杂度和上面一样，但是耗时会更少，消耗内存也更少，因为少了递归栈的处理和记忆化字典的处理两个操作。


### 解答

	class Solution(object):
	    def countHousePlacements(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        if n == 1: return 4
	        if n == 2: return 9
	        dp = [0] * (n+1)
	        dp[1] = 2
	        dp[2] = 3
	        for i in range(3, n+1):
	            dp[i] = dp[i-2] + dp[i-1]
	        return pow(dp[n], 2) % (10 ** 9 + 7)

### 运行结果

	150 / 150 test cases passed.
	Status: Accepted
	Runtime: 192 ms
	Memory Usage: 18.2 MB

### 状态压缩的动态规划

使用动态规划也有两种形式，一种使用列表 dp ，自底向上进行动态规划的常规计算得到最后的结果 dp[n] ，就像上面介绍的一样，另一种我们发现斐波那契数列中，其实当前值只与前前个与前个两个数字有关，所以我这里为了节省空间，使用了压缩状态的动态规划，只需要三个变量即可完成状态的转移。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def countHousePlacements(self, n):
	        """
	        :type n: int
	        :rtype: int
	        """
	        if n == 1: return 4
	        if n == 2: return 9
	        pre, cur = 2, 3
	        for i in range(3, n+1):
	            tmp = pre + cur
	            pre = cur
	            cur = tmp
	        return (cur * cur) % (10**9+7)


### 运行结果

	150 / 150 test cases passed.
	Status: Accepted
	Runtime: 154 ms
	Memory Usage: 14 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-299/problems/count-number-of-ways-to-place-houses/

您的支持是我最大的动力
