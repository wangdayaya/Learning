leetcode 2234. Maximum Total Beauty of the Gardens （python）




### 描述


Alice is a caretaker of n gardens and she wants to plant flowers to maximize the total beauty of all her gardens.

You are given a 0-indexed integer array flowers of size n, where flowers[i] is the number of flowers already planted in the ith garden. Flowers that are already planted cannot be removed. You are then given another integer newFlowers, which is the maximum number of flowers that Alice can additionally plant. You are also given the integers target, full, and partial.

A garden is considered complete if it has at least target flowers. The total beauty of the gardens is then determined as the sum of the following:

* The number of complete gardens multiplied by full.
* The minimum number of flowers in any of the incomplete gardens multiplied by partial. If there are no incomplete gardens, then this value will be 0.

Return the maximum total beauty that Alice can obtain after planting at most newFlowers flowers.


Example 1:


	Input: flowers = [1,3,1,1], newFlowers = 7, target = 6, full = 12, partial = 1
	Output: 14
	Explanation: Alice can plant
	- 2 flowers in the 0th garden
	- 3 flowers in the 1st garden
	- 1 flower in the 2nd garden
	- 1 flower in the 3rd garden
	The gardens will then be [3,6,2,2]. She planted a total of 2 + 3 + 1 + 1 = 7 flowers.
	There is 1 garden that is complete.
	The minimum number of flowers in the incomplete gardens is 2.
	Thus, the total beauty is 1 * 12 + 2 * 1 = 12 + 2 = 14.
	No other way of planting flowers can obtain a total beauty higher than 14.
	
Example 2:

	Input: flowers = [2,4,5,3], newFlowers = 10, target = 5, full = 2, partial = 6
	Output: 30
	Explanation: Alice can plant
	- 3 flowers in the 0th garden
	- 0 flowers in the 1st garden
	- 0 flowers in the 2nd garden
	- 2 flowers in the 3rd garden
	The gardens will then be [5,4,5,5]. She planted a total of 3 + 0 + 0 + 2 = 5 flowers.
	There are 3 gardens that are complete.
	The minimum number of flowers in the incomplete gardens is 4.
	Thus, the total beauty is 3 * 2 + 4 * 6 = 6 + 24 = 30.
	No other way of planting flowers can obtain a total beauty higher than 30.
	Note that Alice could make all the gardens complete but in this case, she would obtain a lower total beauty.






Note:

	1 <= flowers.length <= 10^5
	1 <= flowers[i], target <= 10^5
	1 <= newFlowers <= 10^10
	1 <= full, partial <= 10^5


### 解析

根据题意，爱丽丝是 n 个花园的看守人，她想种花以最大化她所有花园的整体美感。

给定一个大小为 n 的 0 索引整数数组 flowers，其中 flowers[i] 是第 i 个花园中已经种植的花的数量。 已经种下的花不能摘掉。 然后给你另一个整数 newFlowers ，这是 Alice 可以额外种植的花的最大数量。 另外还给了参数 target 、full 和 partial 。

如果某个花园至少有 target 朵花卉，则认为花园是完整的。 然后将所有花园的整体美感值确定为以下各项的总和：

* 完整的花园的数量乘以 full 
* 不完整花园中出现的最小花朵数乘以 partial 。 如果没有不完整的花园，则 partial 为 0

返回 Alice 最多种植 newFlowers 花朵后所能获得的最大整体美感。

这道题很明显就是考察一个贪心的算法，要想使所有花园的整体美感最大，目标为美丽程度 = 不完整花园最小花朵数的最大值 \* partial + 完整花园的个数 \* full 。所以将所有的花园经过排序之后，整体的思路就是在遍历所有的位置，找出使所有花园整体美感最大值出来，细节比较多，具体的代码思路我都放在了注释之中，这一版代码应该是最容易理解的了，我在论坛上也找了好几种解法，但是都太晦涩难懂。

时间复杂度为 O(NlogN)，空间复杂度为 O(N) 。

这位大佬的解法很容易理解，我也是结合大佬的图解更容易理解：https://leetcode.com/problems/maximum-total-beauty-of-the-gardens/discuss/1931085/Python-Explanation-with-pictures-Greedy。

### 解答
				
	class Solution(object):
	    def maximumBeauty(self, flowers, newFlowers, target, full, partial):
	        """
	        :type flowers: List[int]
	        :type newFlowers: int
	        :type target: int
	        :type full: int
	        :type partial: int
	        :rtype: int
	        """
	        flowers = [min(target, a) for a in flowers]
	        # 经过升序排序，目标就变为美丽值=不完整花园最小花朵数的最大值 * partial + 完整花园的个数 * full
	        flowers.sort()
	
	        # 如果 flowers 中最小的都大于等于 target ，说明所有的花园现在都是完整的
	        if min(flowers) == target: return full * len(flowers)
	        # 如果 newFlowers 大于等于将所有的 flowers 都填充为 target 的所需花朵数量
	        # 这种情况下直接比较 max（所有的花园都填为完整的美丽值，将花园填充为只有一个不完整花园的美丽值）
	        if newFlowers >= target * len(flowers) - sum(flowers):
	            return max(full * len(flowers), full * (len(flowers) - 1) + partial * (target - 1))
	
	        # 要想使不完整花园中最大化最小花朵数，就要用花朵填充，使用 cost[i] 计算将前 i 个花园填充为和索引为 i 个花园一样多所需的花朵数
	        cost = [0]
	        for i in range(1, len(flowers)):
	            pre = cost[-1]
	            cost.append(pre + i * (flowers[i] - flowers[i - 1]))
	
	        # 因为有些花园已经达到 target 目标，所以我们要缩小花园计算范围
	        j = len(flowers) - 1
	        while flowers[j] == target:
	            j -= 1
	
	        ans = 0
	        # 遍历寻找最后的答案
	        while newFlowers >= 0:
	            # 表示 idx 及其之前的花园都为不完整花园
	            idx = min(j, bisect_right(cost, newFlowers) - 1)
	            # 将索引 idx 之前的不完整花园都填充和索引为 idx 花园一样的花朵数，并且将剩余的花平均填充到索引为 idx 及其之前的花园中，尽量最大化最小花朵数
	            bar = flowers[idx] + (newFlowers - cost[idx]) // (idx + 1)
	            # 计算出的美丽值结果与结果值进行比较取较大值
	            ans = max(ans, bar * partial + full * (len(flowers) - j - 1))
	            # 现在我们假定已经完成了索引为 j 花园的填充，更新 newFlowers 和 j
	            newFlowers -= (target - flowers[j])
	            j -= 1
	        return ans
            	      
			
### 运行结果

	Runtime: 985 ms, faster than 100.00% of Python online submissions for Maximum Total Beauty of the Gardens.
	Memory Usage: 25.2 MB, less than 17.65% of Python online submissions for Maximum Total Beauty of the Gardens.


### 原题链接

https://leetcode.com/contest/weekly-contest-288/problems/maximum-total-beauty-of-the-gardens/


您的支持是我最大的动力
