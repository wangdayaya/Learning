leetcode  2274. Maximum Consecutive Floors Without Special Floors（python）




### 描述

Alice manages a company and has rented some floors of a building as office space. Alice has decided some of these floors should be special floors, used for relaxation only.

You are given two integers bottom and top, which denote that Alice has rented all the floors from bottom to top (inclusive). You are also given the integer array special, where special[i] denotes a special floor that Alice has designated for relaxation.

Return the maximum number of consecutive floors without a special floor.



Example 1:

	Input: bottom = 2, top = 9, special = [4,6]
	Output: 3
	Explanation: The following are the ranges (inclusive) of consecutive floors without a special floor:
	- (2, 3) with a total amount of 2 floors.
	- (5, 5) with a total amount of 1 floor.
	- (7, 9) with a total amount of 3 floors.
	Therefore, we return the maximum number which is 3 floors.

	
Example 2:

	Input: bottom = 6, top = 8, special = [7,6,8]
	Output: 0
	Explanation: Every floor rented is a special floor, so we return 0.






Note:

	1 <= special.length <= 10^5
	1 <= bottom <= special[i] <= top <= 10^9
	All the values of special are unique.


### 解析

根据题意，Alice 管理一家公司，并租用了建筑物的一些楼层作为办公空间。 爱丽丝决定其中一些楼层应该是特殊楼层，仅用于放松。 给定两个整数 bottom 和 top，表示Alice 已经租下了从 bottom 到 top（包括）的所有楼层。 还给定一个整数数组 special，其中 special[i] 表示 Alice 指定用于放松的特殊楼层。返回没有特殊楼层的最大连续楼层数。


这道题其实就是考察排序，我们先将 special 进行升序排序，然后我们遍历 special 中的所有楼层，不断去计算在指定了放松楼层之间的最大连续楼层数，最后再和 special[-1] 到 top 的最大连续非特殊楼层数进行比较得到最大值，然后再和 bottom 到 special[0]  的最大连续非特殊楼层进行比较得到最大值，最后返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答
				

	class Solution(object):
	    def maxConsecutive(self, bottom, top, special):
	        """
	        :type bottom: int
	        :type top: int
	        :type special: List[int]
	        :rtype: int
	        """
	        result = 0
	        special.sort()
	        for i in range(1, len(special)):
	            result = max(result, special[i]-special[i-1]-1)
	        result = max(result, top-special[-1])
	        result = max(result, special[0]-bottom)
	        return result
            	      
			
### 运行结果


	
	80 / 80 test cases passed.
	Status: Accepted
	Runtime: 915 ms
	Memory Usage: 25.4 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-293/problems/maximum-consecutive-floors-without-special-floors/

您的支持是我最大的动力
