leetcode 2363. Merge Similar Items （python）




### 描述

You are given two 2D integer arrays, items1 and items2, representing two sets of items. Each array items has the following properties:

* items[i] = [valuei, weighti] where valuei represents the value and weighti represents the weight of the ith item.
* The value of each item in items is unique.

Return a 2D integer array ret where ret[i] = [valuei, weighti], with weighti being the sum of weights of all items with value valuei. Note: ret should be returned in ascending order by value.



Example 1:

	Input: items1 = [[1,1],[4,5],[3,8]], items2 = [[3,1],[1,5]]
	Output: [[1,6],[3,9],[4,5]]
	Explanation: 
	The item with value = 1 occurs in items1 with weight = 1 and in items2 with weight = 5, total weight = 1 + 5 = 6.
	The item with value = 3 occurs in items1 with weight = 8 and in items2 with weight = 1, total weight = 8 + 1 = 9.
	The item with value = 4 occurs in items1 with weight = 5, total weight = 5.  
	Therefore, we return [[1,6],[3,9],[4,5]].

	
Example 2:

	Input: items1 = [[1,1],[3,2],[2,3]], items2 = [[2,1],[3,2],[1,3]]
	Output: [[1,4],[2,4],[3,4]]
	Explanation: 
	The item with value = 1 occurs in items1 with weight = 1 and in items2 with weight = 3, total weight = 1 + 3 = 4.
	The item with value = 2 occurs in items1 with weight = 3 and in items2 with weight = 1, total weight = 3 + 1 = 4.
	The item with value = 3 occurs in items1 with weight = 2 and in items2 with weight = 2, total weight = 2 + 2 = 4.
	Therefore, we return [[1,4],[2,4],[3,4]].


Example 3:


	
	Input: items1 = [[1,3],[2,2]], items2 = [[7,1],[2,2],[1,4]]
	Output: [[1,7],[2,4],[7,1]]
	Explanation:
	The item with value = 1 occurs in items1 with weight = 3 and in items2 with weight = 4, total weight = 3 + 4 = 7. 
	The item with value = 2 occurs in items1 with weight = 2 and in items2 with weight = 2, total weight = 2 + 2 = 4. 
	The item with value = 7 occurs in items2 with weight = 1, total weight = 1.
	Therefore, we return [[1,7],[2,4],[7,1]].

Note:

	1 <= items1.length, items2.length <= 1000
	items1[i].length == items2[i].length == 2
	1 <= valuei, weighti <= 1000
	Each valuei in items1 is unique.
	Each valuei in items2 is unique.


### 解析

根据题意，给定两个二维整数数组 items1 和 items2，代表两组项目。 每个数组项具有以下属性：

* items[i] = [valuei, weighti] 其中 valuei 表示值， weighti 表示第 i 个项目的权重。
* items 中每一项的 value 都是唯一的。

返回一个二维整数数组 ret，其中 ret[i] = [valuei, weighti]，其中 weighti 是所有值为 valuei 的项目的权重之和。注意： ret 应该按值升序返回。

这道题其实就是考察字典的常规统计应用，我们分别遍历 items1 和 items2 中的 value 和 weight ，是用字典 d 对 value 的 weight 进行统计，然后将得到的 d 中的键值对变化为列表存到 result 中，最后对 result 进行生序排序返回即可。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。
### 解答


	class Solution(object):
	    def mergeSimilarItems(self, items1, items2):
	        """
	        :type items1: List[List[int]]
	        :type items2: List[List[int]]
	        :rtype: List[List[int]]
	        """
	        d = collections.defaultdict(int)
	        for v, w in items1:
	            d[v] += w
	        for v, w in items2:
	            d[v] += w
	        result = []
	        for k, v in d.items():
	            result.append([k, v])
	        result.sort()
	        return result
### 运行结果

	49 / 49 test cases passed.
	Status: Accepted
	Runtime: 156 ms
	Memory Usage: 14.3 MB

### 解析

因为本身是一个统计的应用题，所以使用 python 的内置函数可以直接进行计算。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。
### 解答


	class Solution(object):
	    def mergeSimilarItems(self, items1, items2):
	        """
	        :type items1: List[List[int]]
	        :type items2: List[List[int]]
	        :rtype: List[List[int]]
	        """
	        d = collections.Counter(dict(items1)) + collections.Counter(dict(items2))
	        return sorted(d.items())
### 运行结果

	
	49 / 49 test cases passed.
	Status: Accepted
	Runtime: 124 ms
	Memory Usage: 14.2 MB


### 原题链接

	https://leetcode.com/contest/biweekly-contest-84/problems/merge-similar-items/


您的支持是我最大的动力
