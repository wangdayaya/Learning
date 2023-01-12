leetcode  1010. Pairs of Songs With Total Durations Divisible by 60（python）

### 每日经典

《望天门山》 ——李白（唐）


天门中断楚江开，碧水东流至此回。

两岸青山相对出，孤帆一片日边来。

### 描述


You are given a list of songs where the i<sub>th</sub> song has a duration of time[i] seconds.

Return the number of pairs of songs for which their total duration in seconds is divisible by 60. Formally, we want the number of indices i, j such that i < j with (time[i] + time[j]) % 60 == 0.


Example 1:

	Input: time = [30,20,150,100,40]
	Output: 3
	Explanation: Three pairs have a total duration divisible by 60:
	(time[0] = 30, time[2] = 150): total duration 180
	(time[1] = 20, time[3] = 100): total duration 120
	(time[1] = 20, time[4] = 40): total duration 60

	
Example 2:

	Input: time = [60,60,60]
	Output: 3
	Explanation: All three pairs have a total duration of 120, which is divisible by 60.







Note:

	1 <= time.length <= 6 * 10^4
	1 <= time[i] <= 500



### 解析

根据题意，给出一个歌曲列表，其中第 i 首歌曲的持续时间为 time[i] 秒。返回总持续时间（以秒为单位）可被 60 整除的歌曲对的数量。形式上，我们想要索引 i 、j 的数量，使得 i < j 且 (time[i] + time[j]) % 60 == 0 。

其实这道题题意很简单，我们在看了限制条件之后发现 time 数组很长，所以暴力解决的方法的时间复杂度为 O(n^2) 是行不通的，但是这里可以给出代码供参考。思路就是先把所有的歌曲时间都对 60 取模存到 time 数组中，然后两层循环遍历数组，只要有 time[i] + time[j] == 60 或者 time[i] + time[j] == 0 ，就将结果加一，遍历结束返回 result 即可，思路很简单，但是不会通过。

### 解答

	
	class Solution(object):
	    def numPairsDivisibleBy60(self, time):
	        """
	        :type time: List[int]
	        :rtype: int
	        """
	        for i,t in enumerate(time):
	            time[i] = t%60
	        result = 0
	        for i,t in enumerate(time):
	            for j in range(i+1,len(time)):
	                if time[i] + time[j] == 60 or time[i] + time[j] == 0:
	                    result += 1
	        return result
### 运行结果

	Time Limit Exceeded
	
	
### 解析

其实上面超时就是因为那个两层循环，我们可以在上面的基础上做一下改进，简化为一层循环，这样时间复杂度会降到 O(n) ，但是要借用字典做一些前期运算。在对 time 取模的时候，对取模的结果也做一个统计存储在字典 d 中，举例：time = [30,20,150,100,40,60,60] ，经过取模就是 time = [30,20,30,40,40,0,0] ， 计算结果 d = {30:2, 20:1,  40:2, 0:2} 。

* 然后遍历字典即可，如果时间 t == 30 或者 t == 0 的歌曲，直接进行计算 d[t] * (d[t] - 1) // 2 加入到 result 中即可，如遍历到上面取模为 30 的歌曲有 2 首，可以合成 1 对，取模为 0 的歌曲同理计算；

* 否则如果 60 - t 在字典中，d[t] * d[60 - t] 加入到 result 中即可 ，同时为了避免重复计算，将 d[t] 和 d[60-t]  都设置为 0 ，如遍历到取模为 20 的歌曲，肯定是可以和取模为 40 的歌曲结合的，总个数为 d[20] * d[40] , 但是此时已经将两个歌曲的可结合对都计算入结果中，所以将值都置为 0 ，避免后面遍历到取模为 40 的歌曲时，重复加入歌曲对；

* 遍历字典结束，将结果 result 返回。




其实一般情况我们发现，有时候算法只要在暴力算法的基础上做些优化就可以通过了。

### 解答
				
	class Solution(object):
	    def numPairsDivisibleBy60(self, time):
	        """
	        :type time: List[int]
	        :rtype: int
	        """
	        d = {}
	        result = 0
	        for i, t in enumerate(time):
	            t = t % 60
	            if t not in d:
	                d[t] = 1
	            else:
	                d[t] += 1
	            time[i] = t
	        for i, t in enumerate(d):
	            if t == 30 or t == 0:
	                c = d[t]
	                result += c * (c - 1) // 2
	            elif 60-t in d:
	                result += (d[t] * d[60 - t])
	                d[t] = 0
	                d[60 - t] = 0
	        return result
            	      
			
### 运行结果

	Runtime: 192 ms, faster than 73.42% of Python online submissions for Pairs of Songs With Total Durations Divisible by 60.
	Memory Usage: 16.2 MB, less than 81.01% of Python online submissions for Pairs of Songs With Total Durations Divisible by 60.


原题链接：https://leetcode.com/problems/pairs-of-songs-with-total-durations-divisible-by-60/



您的支持是我最大的动力
