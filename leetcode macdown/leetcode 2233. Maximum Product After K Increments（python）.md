leetcode  2233. Maximum Product After K Increments（python）




### 描述


You are given an array of non-negative integers nums and an integer k. In one operation, you may choose any element from nums and increment it by 1.

Return the maximum product of nums after at most k operations. Since the answer may be very large, return it modulo 10^9 + 7.


Example 1:

	Input: nums = [0,4], k = 5
	Output: 20
	Explanation: Increment the first number 5 times.
	Now nums = [5, 4], with a product of 5 * 4 = 20.
	It can be shown that 20 is maximum product possible, so we return 20.
	Note that there may be other ways to increment nums to have the maximum product.

	
Example 2:

	Input: nums = [6,3,3,2], k = 2
	Output: 216
	Explanation: Increment the second number 1 time and increment the fourth number 1 time.
	Now nums = [6, 4, 3, 3], with a product of 6 * 4 * 3 * 3 = 216.
	It can be shown that 216 is maximum product possible, so we return 216.
	Note that there may be other ways to increment nums to have the maximum product.



Note:


	1 <= nums.length, k <= 10^5
	0 <= nums[i] <= 10^6

### 解析


根据题意，给定一个非负整数数组 nums 和一个整数 k 。 在一次操作中，可以从 nums 中选择任何一个元素并将其加 1 。

在经过最多 k 次这样的操作后返回 nums 中所有元素的最大乘积。 由于答案可能非常大，因此以 10^9 + 7 为模返回。

这道题很明显就是考察堆的数据结构的用法，因为我用的是 python ，所以有已经封装好的 heapq 模块方便使用。

要想使得最后的乘积变得最大，很关键就是让最小的数字尽可能变大，思路还是很简单的：

* 将 nums 装入堆中
* 我们进行 k 次 while 循环，每次循环，将 nums 中最小的数字弹出，然后将其加一，然后再将其插入 nums 中，同时 k 减一
* 循环结束之后，将所有的元素进行乘法运算，注意每次进行乘法运算之后要进行取模，否则结果大小了，容易超时，不信你可以自己试试
* 最后得到的乘积肯定是最大的，将其返回即可

这道题刚好能够 AC ，因为 nums 的长度为 10^5 ，而堆的弹出时间复杂度为 O(1) ，基本忽略不计，而堆的插入的时间复杂度为  O(logN) ，而 k 最大可能为 10^5 ，所以经过 k 次操作总共的时间复杂度为 O(NlogN) ，空间复杂度为 O(1) 。


### 解答
				

	class Solution(object):
	    def maximumProduct(self, nums, k):
	        """
	        :type nums: List[int]
	        :type k: int
	        :rtype: int
	        """
	        heapq.heapify(nums)
	        result = 1
	        mod = pow(10, 9) + 7
	        while k > 0:
	            a = heapq.heappop(nums)
	            a += 1
	            heapq.heappush(nums, a)
	            k -= 1
	        for n in nums:
	            result = (result * n)% mod
	        return result
            	      
			
### 运行结果


	
	73 / 73 test cases passed.
	Status: Accepted
	Runtime: 6612 ms
	Memory Usage: 22.5 MB

### 原题链接



https://leetcode.com/contest/weekly-contest-288/problems/maximum-product-after-k-increments/


您的支持是我最大的动力
