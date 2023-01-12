leetcode  2322. Minimum Score After Removals on a Tree（python）




### 描述

There is an undirected connected tree with n nodes labeled from 0 to n - 1 and n - 1 edges.

You are given a 0-indexed integer array nums of length n where nums[i] represents the value of the ith node. You are also given a 2D integer array edges of length n - 1 where edges[i] = [ai, bi] indicates that there is an edge between nodes ai and bi in the tree.

Remove two distinct edges of the tree to form three connected components. For a pair of removed edges, the following steps are defined:

* Get the XOR of all the values of the nodes for each of the three components respectively.
* The difference between the largest XOR value and the smallest XOR value is the score of the pair.
* For example, say the three components have the node values: [4,5,7], [1,9], and [3,3,3]. The three XOR values are 4 ^ 5 ^ 7 = 6, 1 ^ 9 = 8, and 3 ^ 3 ^ 3 = 3. The largest XOR value is 8 and the smallest XOR value is 3. The score is then 8 - 3 = 5.

Return the minimum score of any possible pair of edge removals on the given tree.

 



Example 1:


![](https://assets.leetcode.com/uploads/2022/05/03/ex1drawio.png)

	Input: nums = [1,5,5,4,11], edges = [[0,1],[1,2],[1,3],[3,4]]
	Output: 9
	Explanation: The diagram above shows a way to make a pair of removals.
	- The 1st component has nodes [1,3,4] with values [5,4,11]. Its XOR value is 5 ^ 4 ^ 11 = 10.
	- The 2nd component has node [0] with value [1]. Its XOR value is 1 = 1.
	- The 3rd component has node [2] with value [5]. Its XOR value is 5 = 5.
	The score is the difference between the largest and smallest XOR value which is 10 - 1 = 9.
	It can be shown that no other pair of removals will obtain a smaller score than 9.
	
Example 2:


![](https://assets.leetcode.com/uploads/2022/05/03/ex2drawio.png)

	Input: nums = [5,5,2,4,4,2], edges = [[0,1],[1,2],[5,2],[4,3],[1,3]]
	Output: 0
	Explanation: The diagram above shows a way to make a pair of removals.
	- The 1st component has nodes [3,4] with values [4,4]. Its XOR value is 4 ^ 4 = 0.
	- The 2nd component has nodes [1,0] with values [5,5]. Its XOR value is 5 ^ 5 = 0.
	- The 3rd component has nodes [2,5] with values [2,2]. Its XOR value is 2 ^ 2 = 0.
	The score is the difference between the largest and smallest XOR value which is 0 - 0 = 0.
	We cannot obtain a smaller score than 0.





Note:

	n == nums.length
	3 <= n <= 1000
	1 <= nums[i] <= 10^8
	edges.length == n - 1
	edges[i].length == 2
	0 <= ai, bi < n
	ai != bi
	edges represents a valid tree.


### 解析

根据题意，有一棵无向连通树，其 n 个节点标记为从 0 到 n - 1 ，并且有 n - 1 条边。给定一个长度为 n 的 0 索引整数数组 nums ，其中 nums[i] 表示第 i 个节点的值。给定一个长度为 n - 1 的二维整数数组 edges ，其中 edges[i] = [ai, bi] 表示树中节点 ai 和 bi 之间有一条边。

删除树的两个不同边以形成三个不同的连通组件。对于被移除的一对边，定义了以下步骤：

* 分别对三个组件中的每一个的节点值进行异或
* 最大 XOR 值和最小 XOR 值之间的差就是这对边的得分
* 例如，假设将一个树分成三个连通组件，每个组件具有节点值：[4,5,7]、[1,9] 和 [3,3,3]。三个异或值分别为4 ^ 5 ^ 7 = 6、1 ^ 9 = 8、3 ^ 3 ^ 3 = 3。最大的异或值是 8 ，最小的异或值是 3 。那么移除这对边的分数是 8 - 3 = 5

返回给定树上移除任何可能的两条边后的最低分数。

我们在树上任意去掉两个边得到三个树结构，由于只有最多 1000 个节点和边，所以我们可以进行二层循环的遍历来寻找满足题意的两条边，时间复杂度是允许的。所以我们主要的难点在于判断去掉某两条边之后所产生的三个树之间的关系，以及计算每个树所有节点的值的异或分数，经过分析有以下三种情况，假设根节点为 Root ，下面有 A 子树和 B 子树：

*  A 子树和 B 子树是兄弟关系，那么A 子树、 B 子树、Root 三部分分数分别为 XOR[A]、XOR[B]、XOR[Root]^XOR[A]^XOR[B]
*  A 子树是 B 子树的父节点，那么A 子树、 B 子树、Root 三部分的分数分别为 XOR[A]^XOR[B]、XOR[B]、XOR[Root]^XOR[A]
*  B 子树是 A 子树的父节点，那么A 子树、 B 子树、Root 三部分的分数分别为 XOR[A]、XOR[B]^XOR[A] 、XOR[Root]^XOR[B]

还有不懂的同学可以看[灵神的解析过程](https://leetcode.cn/problems/minimum-score-after-removals-on-a-tree/solution/dfs-shi-jian-chuo-chu-li-shu-shang-wen-t-x1kk/)。
时间复杂度为 O(N^2) ，空间复杂度为 O(N) 。




### 解答

	class Solution:
	    def minimumScore(self, nums: List[int], edges: List[List[int]]) -> int:
	        N = len(nums)
	        graph = [[] for _ in range(N)]
	        for x, y in edges:
	            graph[x].append(y)
	            graph[y].append(x)
	
	        xor, in_, out, clock = [0] * N, [0] * N, [0] * N, 0
	
	        def dfs(x, father):
	            nonlocal clock
	            clock += 1
	            in_[x] = clock
	            xor[x] = nums[x]
	            for y in graph[x]:
	                if y != father:
	                    dfs(y, x)
	                    xor[x] ^= xor[y]
	            out[x] = clock
	
	        dfs(0, -1)
	        result = float("inf")
	        for i in range(2, N):
	            for j in range(1, i):
	                if in_[i] < in_[j] <= out[i]:  # i 是 j 的祖先
	                    x, y, z = xor[j], xor[i] ^ xor[j], xor[0] ^ xor[i]
	                elif in_[j] < in_[i] <= out[j]:  # j 是 i 的祖先
	                    x, y, z = xor[i], xor[i] ^ xor[j], xor[0] ^ xor[j]
	                else:  # i 和 j 是兄弟节点
	                    x, y, z = xor[i], xor[j], xor[0] ^ xor[i] ^ xor[j]
	                result = min(result, max(x, y, z) - min(x, y, z))
	                if result == 0:
	                    return 0
	        return result

### 运行结果

	
	65 / 65 test cases passed.
	Status: Accepted
	Runtime: 2470 ms
	Memory Usage: 16.1 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-299/problems/minimum-score-after-removals-on-a-tree/


您的支持是我最大的动力
