leetcode  1354. Construct Target Array With Multiple Sums（python）




### 描述

You are given an array target of n integers. From a starting array arr consisting of n 1's, you may perform the following procedure :

* let x be the sum of all elements currently in your array.
* choose index i, such that 0 <= i < n and set the value of arr at index i to x.
* You may repeat this procedure as many times as needed.

Return true if it is possible to construct the target array from arr, otherwise, return false.

 



Example 1:

	Input: target = [9,3,5]
	Output: true
	Explanation: Start with arr = [1, 1, 1] 
	[1, 1, 1], sum = 3 choose index 1
	[1, 3, 1], sum = 5 choose index 2
	[1, 3, 5], sum = 9 choose index 0
	[9, 3, 5] Done

	
Example 2:


	Input: target = [1,1,1,2]
	Output: false
	Explanation: Impossible to create target array from [1,1,1,1].

Example 3:

	Input: target = [8,5]
	Output: true





Note:

	
	n == target.length
	1 <= n <= 5 * 10^4
	1 <= target[i] <= 10^9

### 解析

根据题意，给定一个包含 n 个整数的数组 target 。 从包含 n 个 1 的起始数组 arr 中，可以执行以下过程：

* 当前数组中所有元素的总和为 x 
* 选择索引 i ，使得 0 <= i < n 并将索引 i 处的 arr 值设置为 x 。
* 您可以根据需要多次重复此过程。

如果可以从 arr 构造目标数组，则返回 true ，否则返回 false 。

其实就是模拟倒推整个的 A 能否回到 [1,1,...,1] 的初始状态，这里需要注意的是很多边界条件，还有需要特别注意优化的一步计算过程，因为碰到 [1,100000001] 这类用例肯定会超时，因为计算步骤太多，这里我们用到取模的操作，如果某个位置是最大的数，但是其在进行好几次逆操作之后仍然是最大的数，那么我们就能够用取模来进行计算的加速。如下的例子，

	25，3，5
	17，3，5
	9，3，5
	1，3，5
	1，3，1
	1，1，1

我们可以直接使用取模直接从第一步跳到第四步，节省很多时间。


时间复杂度为 O(NlogN) ，空间复杂度为 O(N)  。

### 解答
				

	class Solution(object):
	    def isPossible(self, A):
	        """
	        :type target: List[int]
	        :rtype: bool
	        """
	        total = sum(A)
	        target = [-i for i in A]  # heap q默认的是小根堆 这边加负号就相当于变成了一个大根堆
	        heapq.heapify(target)  
	        while True:
	            max = -heapq.heappop(target)  # 取最大数
	            if max == 1:  # 说明此刻 target 里面全都是 -1 了
	                return True
	            if 2 * max - total < 1:  # 如果是 [4,2,2] 这种一旦逆操作一次变成 [0,2,2] ，已经不符合题意了，以为出现了小于 1 的数
	                return False
	            sub = total - max  #  sub 表示除 max 以外所有数字的和
	            new = max % sub or sub  # 如果出现[1,1000000001] 极端用例， 因为第二个位置的数字在很多次的次逆操作都是最大的，每次都需要减 sub 所以干脆直接取模，但同整除的情况下是不能直接 new=0 的 所以会是sub
	            total -= max - new  # 更新 total ，从 total 中去掉 max 直接降到 new 的差值
	            heapq.heappush(target, -new)
	            
### 运行结果

	
	71 / 71 test cases passed.
	Status: Accepted
	Runtime: 259 ms
	Memory Usage: 18 MB


### 原题链接

https://leetcode.com/problems/construct-target-array-with-multiple-sums/


您的支持是我最大的动力
