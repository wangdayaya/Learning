leetcode 923. 3Sum With Multiplicity （python）




### 描述


Given an integer array arr, and an integer target, return the number of tuples i, j, k such that i < j < k and arr[i] + arr[j] + arr[k] == target.

As the answer can be very large, return it modulo 10^9 + 7.

 


Example 1:

	
	Input: arr = [1,1,2,2,3,3,4,4,5,5], target = 8
	Output: 20
	Explanation: 
	Enumerating by the values (arr[i], arr[j], arr[k]):
	(1, 2, 5) occurs 8 times;
	(1, 3, 4) occurs 8 times;
	(2, 2, 4) occurs 2 times;
	(2, 3, 3) occurs 2 times.
	
Example 2:


	Input: arr = [1,1,2,2,2,2], target = 5
	Output: 12
	Explanation: 
	arr[i] = 1, arr[j] = arr[k] = 2 occurs 12 times:
	We choose one 1 from [1,1] in 2 ways,
	and two 2s from [2,2,2,2] in 6 ways.



Note:


	3 <= arr.length <= 3000
	0 <= arr[i] <= 100
	0 <= target <= 300


### 解析

根据题意，给定一个整数数组 arr 和一个整数 target ，返回三元组 i、j、k 的数量，使得 i < j < k 并且 arr[i] + arr[j] + arr[k] == target。由于答案可能非常大，因此以 10^9 + 7 为模返回。

这道题的题意说的是十分简洁明了，但是解决起来有点麻烦，我开始看了例子一觉得挺简单的，但是当我写了一半的代码看到例子二的时候，我就不淡定了，这道题虽然是考察排列组合，但是还是有一些坑在里面的，比如可能 a 、b 、c 三个数字中有可能三个都是相同的数字，也有可能其中两个是相同的数字，或者三个都不相同，所以我们要分不同的情况进行解决。

结合限制条件我们知道 arr 最大长度为 3000 ，arr[i] 最大为 100 ，所以我们知道这道题是允许时间复杂度为 O(N^2）的，过程如下：

* 首先用 count 来对所有 arr 中出现的元素进行计数
* 然后双重循环找可能的 a 和 b ，这样 c 也就为确定的 target - a - b ，如果 c 不合法就直接进行下一次循环；如果 a、b、c 三个数字相同，那么其实就是对 arr 中的元素 a 进行排列组合即可；如果有 a、b、c 中有两个相同的数字，那么其实就是对 arr 中的元素 a 进行排列组合的数量再乘 c 的个数即可；如果  a、b、c 都不相同，那么就是 arr 中的三个元素出现的个数的乘积；
* 最后返回对 10 ** 9 + 7 取模的结果即可

时间复杂度为 O(N^2) ，空间复杂度为 O(N)。


### 解答
				

	class Solution(object):
	    def threeSumMulti(self, arr, target):
	        """
	        :type arr: List[int]
	        :type target: int
	        :rtype: int
	        """
	        mod = 10 ** 9 + 7
	        result = 0
	        cnt = [0]*101
	        for n in arr:
	            cnt[n] += 1
	        for a in range (0,101):
	            for b in range (a,101):
	                c = target - a - b
	                if c < 0 or c > 100 :
	                    continue
	                elif a == b and b == c:
	                    result += (cnt[a] * (cnt[a]-1) * (cnt[a]-2) )/ 6
	                elif a == b and b != c:
	                    result += (cnt[a] * (cnt[a]-1) ) / 2 * cnt[c]
	                elif a < b and b < c :
	                    result += cnt[a] * cnt[b] * cnt[c]
	        return int(result % mod)
	            
	      
			
### 运行结果
	
	Runtime: 84 ms, faster than 63.16% of Python online submissions for 3Sum With Multiplicity.
	Memory Usage: 13.6 MB, less than 36.84% of Python online submissions for 3Sum With Multiplicity.

### 原题链接


https://leetcode.com/problems/3sum-with-multiplicity/


您的支持是我最大的动力
