leetcode  881. Boats to Save People（python）




### 描述

You are given an array people where people[i] is the weight of the i<sub>th</sub> person, and an infinite number of boats where each boat can carry a maximum weight of limit. Each boat carries at most two people at the same time, provided the sum of the weight of those people is at most limit.Return the minimum number of boats to carry every given person.

 



Example 1:

	Input: people = [1,2], limit = 3
	Output: 1
	Explanation: 1 boat (1, 2)

	



Note:

	1 <= people.length <= 5 * 10^4
	1 <= people[i] <= limit <= 3 * 10^4


### 解析


根据题意，给定数组 people ，其中 people[i] 是第 i 个人的重量，以及给出了一个无限个船，每艘船可以承载最大重量 limit 。 每艘船最多同时载运两个人，前提是这些人的重量之和最多为 limit 。返回运送给定 people 的最少船只数量。

这道题很明显又是一道考察贪心算法的经典题目，题目要求在一个床上最多放两个人，并且两个人的体重不超过 limit ，用我们最朴素的思想，首先我们拿出一艘新的船，肯定是尽量先把最胖的人放上去，然后再看能不能把轻的人塞进还有剩余足够空间的船中，这样才能保证用到的船的数量是最少的。

* 首先我们就将 people 进行升序排序
* 然后定义前后两个指针 i 和 j ，在 while 循环中分别向中间移动，i 指针指向的是体重尽量轻的人， j 指针指向的是体重尽量重的人，因为胖的人肯定要占一条船，所以 result 肯定要加一，然后判断 people[i] 和 people[j] 的和如果小于等于 limit ，那么就说明能将最轻的人塞进该船里，将 i 加一去找下一个体重轻的人；否则说明塞不进去，船里只能放一个人，然后 j 减一去找下一个比较重的人，继续进行上述的循环，直到 j > i 跳出循环
* 循环结束，直接返回 result 即可

时间复杂度为 O(N) ，空间复杂度为 O(1) 。
### 解答
				

	class Solution(object):
	    def numRescueBoats(self, people, limit):
	        """
	        :type people: List[int]
	        :type limit: int
	        :rtype: int
	        """
	        result = 0
	        people.sort()
	        i = 0
	        j = len(people) -1 
	        while i <= j:
	            result += 1
	            if people[i] + people[j] <= limit:
	                i += 1
	            j -= 1
	        return result
            
            
            	      
			
### 运行结果


	Runtime: 808 ms, faster than 9.67% of Python online submissions for Boats to Save People.
	Memory Usage: 18.7 MB, less than 65.81% of Python online submissions for Boats to Save People.

### 原题链接

https://leetcode.com/problems/boats-to-save-people/


### 后记

您的支持是我最大的动力
