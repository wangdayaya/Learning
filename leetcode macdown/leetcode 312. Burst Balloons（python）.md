leetcode  312. Burst Balloons（python）

### 每日经典

《春望》 ——杜甫（唐）

国破山河在，城春草木深。感时花溅泪，恨别鸟惊心。

烽火连三月，家书抵万金。白头搔更短，浑欲不胜簪。

### 描述


You are given n balloons, indexed from 0 to n - 1. Each balloon is painted with a number on it represented by an array nums. You are asked to burst all the balloons.

If you burst the ith balloon, you will get nums[i - 1] * nums[i] * nums[i + 1] coins. If i - 1 or i + 1 goes out of bounds of the array, then treat it as if there is a balloon with a 1 painted on it.

Return the maximum coins you can collect by bursting the balloons wisely.


Example 1:

	Input: nums = [3,1,5,8]
	Output: 167
	Explanation:
	nums = [3,1,5,8] --> [3,5,8] --> [3,8] --> [8] --> []
	coins =  3*1*5    +   3*5*8   +  1*3*8  + 1*8*1 = 167

	
Example 2:


	Input: nums = [1,5]
	Output: 10




Note:

	n == nums.length
	1 <= n <= 500
	0 <= nums[i] <= 100


### 解析

根据题意，给出 n 个气球，索引从 0 到 n - 1 。每个气球上都涂有一个数字，由数组 nums 表示。 要求爆破所有的气球。如果打破第 i 个气球，会得到 nums[i - 1] \* nums[i] \* nums[i + 1] 个硬币。 如果 i - 1 或 i + 1 超出数组范围，则将其视为有一个涂有 1 的气球。通过爆破气球来返回可以收集的最大硬币。

这道题是典型的动态规划题目，题目很好理解，关键的难点在于，打了某个气球之后，剩下气球的分值会发生变化，这个过程如果正向去理解会比较有难度，我们可以反向来推这个过程。假如定义动态数组 DP[i][j] 表示在 nums 中的 i 到 j 的闭区间内的气球我们都打爆之后，可以得到的最大的分值，举例 nums = [3,1,5,8] ，当我们在长度为 1 的子数组中打气球可以得到的最大分数为

	在子数组 [3] 中当最后一枪打在 3 的时候 DP[0][0] = 1*3*1 = 3 
	在子数组 [1] 中当最后一枪打在 1 的时候 DP[1][1] = 3*1*5 = 15
	在子数组 [5] 中当最后一枪打在 5 的时候 DP[2][2] = 1*5*8 = 40
	在子数组 [8] 中当最后一枪打在 8 的时候 DP[3][3] = 5*8*1 = 40

当我们在长度为 2 的子数组中打气球可以得到的最大分数为

	在子数组 [3,1] 中当最后一枪打在 3 的时候，也就是先打 1 ，再打 3 ；或者最后一枪打在 1 的时候，也即是先打 3 ，再打 1 , DP[0][1] = max(3*1*5+1*3*5, 1*3*1+1*1*5) = 30  ，也就是最后一枪打在索引为 0 的时候分值最大
	在子数组 [1,5] 中当最后一枪打在 1 时候; 或者最后一枪打在 5 , 和上面同样的道理 DP[1][2] = max(1*5*8+3*1*8, 3*1*5+3*5*8) = 135 ，也就是最后一枪打在索引为 2 的时候分值最大 
	在子数组 [5,8] 中当最后一枪打在 5 的时候；或者最后一枪打在 8 的时候，和上面的道理一样 DP[2][3] = max(5*8*1+1*5*1, 1*5*8+1*8*1) = 48 ，也就是最后一枪打在索引为 3 的时候分值最大
	
当我们在长度为 3 的子数组中打气球可以得到的最大分值为

	当在子数组 [3,1,5] 当中最后一枪打在了 3 时候，说明 [1,5] 已经打完了我们知道得到的最大分数为 135 ；如果最后一枪打在了 1  的时候，说明 [3] 和 [5]  已经打完了，我们已经知道了 [3] 和 [5] 的最大分数分别为 3 和 40 ；如果最后一枪打在了 5 的时候，说明 [3,1] 已经打完了，我们已经知道了 [3,1] 的最大分数为 15 ，此时 DP[0][2] = max(135+1*3*8, 3+40+1*1*8 , 30+1*5*8) = 159 ，也就是最后一枪打在索引 0 的时候分值最大
	当在子数组 [1,5,8] 当中最后一枪打在了 1 的时候，说明 [5,8] 已经打完了我们知道得到的最大分数为 48 ；如果最后一枪打在了 5 时候，说明 [3] 和 [5]  已经打完了，我们已经知道了 [3] 和 [5] 的最大分数分别为 15 和 40 ；如果最后一枪打在了 8 的时候，说明 [1,5] 已经打完了，我们已经知道了 [1,5] 的最大分数为 135 ，此时 DP[1][3] = max(48+3*1*1, 15+40+3*5*1 , 135+3*8*1) = 159 ，也就是最后一枪打在索引 3 的时候分值最大

当我们在长度为 4 的子数组中打气球可以得到的最大分数为

	当子数组 [3,1,5,8] 中最后一枪打在 3 ，说明 [1,5,8] 已经打完了，最大分数为 159 ；最后一枪打在 1 ，说明 [3] 和 [5,8] 已经打完了，我们知道   [3] 和 [5,8]  拿到的最大分数为 3 和 48；最后一枪打在 5 ，说明 [3,1] 和 [8] 已经打完了，我们知道  [3,1] 和 [8]  拿到的最大分数为 30 和 40 ；最后一枪打在 8 ，说明 [3,1,5] 已经打完了，我们知道  [3,1,5]  的最大分数为 159 , DP[0][3] = max(159+1*3*1 , 3+48+1*1*1 ,  30+40+1*5*1 , 159 + 1*8*1 ) = 167 , 也就是最后一枪打在索引为 3的时候分值最大

最后得到的动态规划结果如下表：

| |  0  | 1|2| 3 |
|  ---- |  ----  | ----  | ----  | ----  |
| 0|  3, 0  |    30，0      |    159，0    |    167,3   | 
|  1 |       |  15, 1   |  135，2     |     159，3  |
| 2|        |          |  40, 2 |    48，3   |
|  3 |      |           |       |    40, 3   |

如果题目还要进步求打气球的顺序，我们可以根据上面的表中的记录的索引知道打气球的倒顺序为 3 ，0 ，2 ，1 ，所以最终打气球的索引顺序结果为： 1，2，0，3 。

动态规划公式为：

	for k in range(i,j+1):
    	dp[i][j] = max(dp[i][j], dp[i][k-1]+nums[i-1]*nums[k]*nums[j+1]+dp[k+1][j])

### 解答
				
	class Solution(object):
	    def maxCoins(self, nums):
	        """
	        :type nums: List[int]
	        :rtype: int
	        """
	        N = len(nums)
	        nums = [1]+nums+[1]        
	        dp = [[0 for _ in range(N+2)] for _ in range(N+2)]        
	        for i in range(1,N+1):
	            dp[i][i]=nums[i-1]*nums[i]*nums[i+1]  
	        for Len in range(2,N+1):
	            for i in range(1,N-Len+2):
	                left,right = i,i+Len-1
	                for k in range(left,right+1):
	                    dp[left][right] = max(dp[left][right],dp[left][k-1]+nums[left-1]*nums[k]*nums[right+1]+dp[k+1][right])
	                        
	        return dp[1][N]
	                

            	      
			
### 运行结果


	Runtime: 8042 ms, faster than 33.33% of Python online submissions for Burst Balloons.
	Memory Usage: 18.9 MB, less than 15.05% of Python online submissions for Burst Balloons.

原题链接：https://leetcode.com/problems/burst-balloons/



您的支持是我最大的动力
