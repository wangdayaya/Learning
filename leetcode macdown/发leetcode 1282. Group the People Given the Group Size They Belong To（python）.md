leetcode  1282. Group the People Given the Group Size They Belong To（python）

### 描述


There are n people that are split into some unknown number of groups. Each person is labeled with a unique ID from 0 to n - 1.

You are given an integer array groupSizes, where groupSizes[i] is the size of the group that person i is in. For example, if groupSizes[1] = 3, then person 1 must be in a group of size 3.

Return a list of groups such that each person i is in a group of size groupSizes[i].

Each person should appear in exactly one group, and every person must be in a group. If there are multiple answers, return any of them. It is guaranteed that there will be at least one valid solution for the given input.


Example 1:

	
	Input: groupSizes = [3,3,3,3,3,1,3]
	Output: [[5],[0,1,2],[3,4,6]]
	Explanation: 
	The first group is [5]. The size is 1, and groupSizes[5] = 1.
	The second group is [0,1,2]. The size is 3, and groupSizes[0] = groupSizes[1] = groupSizes[2] = 3.
	The third group is [3,4,6]. The size is 3, and groupSizes[3] = groupSizes[4] = groupSizes[6] = 3.
	Other possible solutions are [[2,1,6],[5],[0,4,3]] and [[5],[0,6,2],[4,3,1]].
	
Example 2:

	Input: groupSizes = [2,1,3,3,3,2]
	Output: [[1],[0,5],[2,3,4]]



Note:

	groupSizes.length == n
	1 <= n <= 500
	1 <= groupSizes[i] <= n



### 解析


根据题意，有 n 个人被分成不知数量的几个组，每个人的唯一 ID 标记就是索引 0 到 n-1 ，每个人都应该恰好出现在一个组中，并且每个人都必须在一个组中。

然后又给了一个整数列表 groupSizes ，其中 groupSizes[i] 表示的是索引为 i 的人所在组的大小。例如，如果 groupSizes[1] = 3，则索引为 1 的人必须在大小为 3 的组中。

要求返回一个满足题意的列表，如果有多个答案，则返回其中任何一个。

其实这个题就是有点绕，理清了题意就好做了，这里主要是借用字典来解题：

* 初始化结果 result 为空列表存放答案， d 为空字典用来存放不同的 size 对应的索引列表，如例子一中得到的中间答案就是： 
	
		d = {1: [5], 3: [0, 1, 2, 3, 4, 6]}
	
* 然后遍历 d 中的 k,v 对，v 是列表，且列表的长度肯定是 k 的整数倍，所以用下面的语句截取长度为 k 的列表放入 result 中：

        for i in range(0,len(v),k):
            result.append(v[i:i+k])
                
* 遍历最后得到的 result 就是答案。   

	
	
### 解答
				

	class Solution(object):
	    def groupThePeople(self, groupSizes):
	        """
	        :type groupSizes: List[int]
	        :rtype: List[List[int]]
	        """
	        result = []
	        d = {}
	        c = collections.Counter(groupSizes)
	        for k,v in c.items():
	            d[k] = []
	        for i, size in enumerate(groupSizes):
	                d[size].append(i)
	        for k,v in d.items():
	            for i in range(0,len(v),k):
	                result.append(v[i:i+k])
	        return result
	        
            	      
			
### 运行结果

	Runtime: 68 ms, faster than 46.94% of Python online submissions for Group the People Given the Group Size They Belong To.
	Memory Usage: 13.4 MB, less than 99.32% of Python online submissions for Group the People Given the Group Size They Belong To.



原题链接：https://leetcode.com/problems/group-the-people-given-the-group-size-they-belong-to/



您的支持是我最大的动力
