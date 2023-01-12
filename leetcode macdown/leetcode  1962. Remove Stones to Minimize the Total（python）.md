leetcode  1962. Remove Stones to Minimize the Total（python）




### 描述

You are given a 0-indexed integer array piles, where piles[i] represents the number of stones in the i<sup>th</sup> pile, and an integer k. You should apply the following operation exactly k times:

* Choose any piles[i] and remove floor(piles[i] / 2) stones from it.

Notice that you can apply the operation on the same pile more than once. Return the minimum possible total number of stones remaining after applying the k operations. floor(x) is the greatest integer that is smaller than or equal to x (i.e., rounds x down).



Example 1:

	Input: piles = [5,4,9], k = 2
	Output: 12
	Explanation: Steps of a possible scenario are:
	- Apply the operation on pile 2. The resulting piles are [5,4,5].
	- Apply the operation on pile 0. The resulting piles are [3,4,5].
	The total number of stones in [3,4,5] is 12.

	
Example 2:

	Input: piles = [4,3,6,7], k = 3
	Output: 12
	Explanation: Steps of a possible scenario are:
	- Apply the operation on pile 2. The resulting piles are [4,3,3,7].
	- Apply the operation on pile 3. The resulting piles are [4,3,3,4].
	- Apply the operation on pile 0. The resulting piles are [2,3,3,4].
	The total number of stones in [2,3,3,4] is 12.





Note:

	1 <= piles.length <= 10^5
	1 <= piles[i] <= 10^4
	1 <= k <= 10^5


### 解析

根据题意，给定一个 0 索引的整数数组 piles ，其中 piles[i] 表示第 i 堆中的石头数量，还给了一个整数 k 。要求我们应该准确进行 k 次以下的操作：

* 选择任何一个 piles[i] 并从中移除 floor(piles[i] / 2)  个石头。

请注意，我们可以多次对同一桩应用该操作。返回应用 k 操作后 piles 中剩余的所有石头可能的最小总数。floor(x) 是小于或等于 x 的最大整数，即向下舍入取整 x 。

这道题一看就是在考察贪心思想的，最朴素、最简单的解决思路就是进行排序找到 piles 中的最大的值，然后将其进行“移除 floor(piles[i] / 2)  个石头”的操作，然后再排序找最大值，重复执行上面的操作，通过 k 次的操作，留下来的 piles 数组中的石头。

但是我们通过观察对于 piles 的限制条件，知道 piles 的长度最长为 10^5  ，每个 piles[i] 最大为 10^4 ，k 最大值为 10^5 ，所以按照上面的方法肯定是超时的。基本的思路是不变的的，但是实现的方式要发生一些转换。k 次的执行肯定是要进行的，所以时间复杂度为 O(N) ，要想 AC 必须要保证整个代码的时间复杂度不能超过 O(NlogN) ，所以每次在执行上面提到的“排序和移除 floor(piles[i] / 2)  个石头”这两个操作的时间要不超过 O(logN) ，这么一来我们就优先选用大根堆数据结构了。

具体实现就是，我们将 piles 中的所有值都放入大根堆中，然后遍历 k 次，每次弹出当前的根结点也就是最大值，将其执行“移除 floor(piles[i] / 2)  个石头”的操作，然后再将剩余的石头加入大根堆中，重复这个过程直到 k 次操作执行完，结果肯定是符合题意的最少石头总数，这样一来在时间复杂度上和空间复杂度上就都满足了，能保证正常的 AC 。

时间复杂度为 O(N+NlogN) ，因为将数组 piles 转换成大根堆需要 O(N) 的时间，当然了也可以简写成 OO(NlogN)，空间复杂度为 O(N) 。



### 解答

	class Solution(object):
	    def minStoneSum(self, piles, k):
	        """
	        :type piles: List[int]
	        :type k: int
	        :rtype: int
	        """
	        h = [-num for num in piles]
	        heapq.heapify(h)
	        for _ in range(k):
	            pile = -heapq.heappop(h)
	            removed = pile // 2
	            heapq.heappush(h, -(pile - removed))
	        return -sum(h)

### 运行结果

* Runtime 3797 ms，Beats 88.37%
* Memory 26 MB，Beats 37.21%

### 原题链接

https://leetcode.com/problems/remove-stones-to-minimize-the-total/


您的支持是我最大的动力
