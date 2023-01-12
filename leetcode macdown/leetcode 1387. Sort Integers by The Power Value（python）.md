leetcode 1387. Sort Integers by The Power Value （python）

### 描述

The power of an integer x is defined as the number of steps needed to transform x into 1 using the following steps:

* if x is even then x = x / 2
* if x is odd then x = 3 * x + 1

For example, the power of x = 3 is 7 because 3 needs 7 steps to become 1 (3 --> 10 --> 5 --> 16 --> 8 --> 4 --> 2 --> 1).

Given three integers l<sub>o</sub>, h<sub>i</sub> and k. The task is to sort all integers in the interval [l<sub>o</sub>, h<sub>i</sub>] by the power value in ascending order, if two or more integers have the same power value sort them by ascending order.

Return the k-th integer in the range [l<sub>o</sub>, h<sub>i</sub>] sorted by the power value.

Notice that for any integer x (l<sub>o</sub> <= x <= h<sub>i</sub>) it is guaranteed that x will transform into 1 using these steps and that the power of x is will fit in 32 bit signed integer.





Example 1:

	Input: lo = 12, hi = 15, k = 2
	Output: 13
	Explanation: The power of 12 is 9 (12 --> 6 --> 3 --> 10 --> 5 --> 16 --> 8 --> 4 --> 2 --> 1)
	The power of 13 is 9
	The power of 14 is 17
	The power of 15 is 17
	The interval sorted by the power value [12,13,14,15]. For k = 2 answer is the second element which is 13.
	Notice that 12 and 13 have the same power value and we sorted them in ascending order. Same for 14 and 15.

	
Example 2:


	Input: lo = 1, hi = 1, k = 1
	Output: 1

Example 3:

	
	Input: lo = 7, hi = 11, k = 4
	Output: 7
	Explanation: The power array corresponding to the interval [7, 8, 9, 10, 11] is [16, 3, 19, 6, 14].
	The interval sorted by power is [8, 10, 11, 7, 9].
	The fourth number in the sorted array is 7.
	
Example 4:


	Input: lo = 10, hi = 20, k = 5
	Output: 13
	
Example 5:


	Input: lo = 1, hi = 1000, k = 777
	Output: 570

Note:

	1 <= lo <= hi <= 1000
	1 <= k <= hi - lo + 1


### 解析

根据题意，就是给出了一个计算一个数字的 power 的方法，那就是将一个数字 x 变为 1
需要经过下面操作的次数：

* x 是偶数则 x/=2
* x 是奇数则 x = 3*x+1

然后给出了范围 [lo, hi] 和 k ，要求我们按幂值升序对区间 [lo, hi] 内的所有整数进行排序，如果两个或多个整数具有相同的幂值，则按升序对它们进行排序。返回按幂值排序的 [lo, hi] 范围内的第 k 个整数。题目保证所有证书都能通过上面的运算从 x 变为 1 。

最简单的方法就是按照题意，写一个获取一个数的 power 值的函数 getPower 。然后将所有整数按照 power 值排列，最后返回第 k 个整数即可。


### 解答
				

	class Solution(object):
	    def getKth(self, lo, hi, k):
	        """
	        :type lo: int
	        :type hi: int
	        :type k: int
	        :rtype: int
	        """
	        def getPower(n):
	            result = 0
	            while n!=1:
	                if n%2==0:
	                    n = n//2
	                else:
	                    n = n * 3 + 1
	                result += 1
	            return result
	        d = []
	        for i in range(lo,hi+1):
	            d.append([i,getPower(i)])
	        d = sorted(d, key=lambda x: x[1])
	        return d[k-1][0]
	            
            	      
			
### 运行结果

	Runtime: 708 ms, faster than 35.66% of Python online submissions for Sort Integers by The Power Value.
	Memory Usage: 13.6 MB, less than 86.82% of Python online submissions for Sort Integers by The Power Value.

### 解析

上面的解法从  [lo, hi]  一个一个计算的过程中有很多对数字的 power 值的过程都是重复的，可以用字典 d 直接记录已经知道 power 的数字，这样在后面计算过程中可以省去很多计算的时间。


### 解答


	class Solution(object):
	    def getKth(self, lo, hi, k):
	        """
	        :type lo: int
	        :type hi: int
	        :type k: int
	        :rtype: int
	        """
	        d = {1: 1}
	        def getPower(n):
	            if n not in d:
	                if n % 2 == 0:
	                    d[n] = 1 + getPower(n / 2)
	                else:
	                    d[n] = 1 + getPower(3*n + 1)
	            return d[n]
	        return sorted((getPower(i), i) for i in range(lo, hi+1))[k-1][1]
	            

### 运行结果

	Runtime: 164 ms, faster than 95.35% of Python online submissions for Sort Integers by The Power Value.
	Memory Usage: 42.7 MB, less than 11.63% of Python online submissions for Sort Integers by The Power Value.

原题链接：https://leetcode.com/problems/sort-integers-by-the-power-value/



您的支持是我最大的动力
