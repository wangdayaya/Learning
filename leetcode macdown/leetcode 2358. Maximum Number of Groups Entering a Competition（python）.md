leetcode  2358. Maximum Number of Groups Entering a Competition（python）




### 描述

You are given a positive integer array grades which represents the grades of students in a university. You would like to enter all these students into a competition in ordered non-empty groups, such that the ordering meets the following conditions:

* The sum of the grades of students in the ith group is less than the sum of the grades of students in the (i + 1)th group, for all groups (except the last).
* The total number of students in the ith group is less than the total number of students in the (i + 1)th group, for all groups (except the last).

Return the maximum number of groups that can be formed.



Example 1:

	
	Input: grades = [10,6,12,7,3,5]
	Output: 3
	Explanation: The following is a possible way to form 3 groups of students:
	- 1st group has the students with grades = [12]. Sum of grades: 12. Student count: 1
	- 2nd group has the students with grades = [6,7]. Sum of grades: 6 + 7 = 13. Student count: 2
	- 3rd group has the students with grades = [10,3,5]. Sum of grades: 10 + 3 + 5 = 18. Student count: 3
	It can be shown that it is not possible to form more than 3 groups.
	
Example 2:

	Input: grades = [8,8]
	Output: 1
	Explanation: We can only form 1 group, since forming 2 groups would lead to an equal number of students in both groups.




Note:

	1 <= grades.length <= 10^5
	1 <= grades[i] <= 10^5


### 解析

根据题意，给定一个正整数数组 grades ，它代表一所大学学生的成绩。 我们希望将所有这些学生按有序的非空组进行比赛，使得排序满足以下条件：

* 对于所有组（最后一组除外），第 i 组学生的成绩总和小于第 (i + 1) 组学生的成绩总和。
* 对于所有组（最后一个除外），第 i 组的学生总数小于第 (i + 1) 组的学生总数。

返回可以形成的最大组数。

其实这道题看起来很复杂，但是具有一定的迷惑性，我们按照贪心的思想知道，想要按照两个条件形成的组数最多，那么肯定要把 grades 进行生序排序，然后第一组取第一个分数，第二组取第二个、第三个分数，第三组取第四个、第五个、第六个分数，以此类推，能形成几组就安排几组，最后返回答案即可。

时间复杂度为 O(NlogN)，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def maximumGroups(self, grades):
	        """
	        :type grades: List[int]
	        :rtype: int
	        """
	        grades.sort()
	        n = 1
	        result = 0
	        while grades:
	            t = copy.deepcopy(n)
	            for _ in range(t):
	                if grades:
	                    grades.pop(0)
	                    t -= 1
	                else:
	                    break
	            if t == 0:
	                result += 1
	                n += 1
	            else:
	                break
	        return result

### 运行结果

	
	68 / 68 test cases passed.
	Status: Accepted
	Runtime: 4341 ms
	Memory Usage: 22.4 MB
	
	
### 解析

上面的写法比较啰嗦，我们可以换一个更简单的思路，我们已经知道了想要构成 n 组，就需要 n(n+1)//2 个人的成绩才行，而我们手上只有 N 个成绩，所以我们先构造出一个数组 L , L[i] 就是构造 i 个组需要的成绩数量，然后我们使用手上的 N 找到 L 中大于等于 N 的数字的索引 i ，如果 L[i] 刚好等于 N 说明刚好能组成 i 组直接返回即可，如果 L[i] 大于 N 说明不能组成 i 组返回 i-1 即可。

时间复杂度为 O(logN) ，空间复杂度为 O(N) 。

### 解答

	class Solution(object):
	    def maximumGroups(self, grades):
	        """
	        :type grades: List[int]
	        :rtype: int
	        """
	        N = len(grades)
	        L = [n*(n+1)//2 for n in range(10001)]
	        i = bisect.bisect_left(L, N)
	        if L[i] == N:
	            return i
	        return i-1
### 运行结果

	68 / 68 test cases passed.
	Status: Accepted
	Runtime: 752 ms
	Memory Usage: 24.2 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-304/problems/maximum-number-of-groups-entering-a-competition/


您的支持是我最大的动力
