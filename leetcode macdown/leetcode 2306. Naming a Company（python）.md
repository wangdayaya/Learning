leetcode 2306. Naming a Company （python）




### 描述

You are given an array of strings ideas that represents a list of names to be used in the process of naming a company. The process of naming a company is as follows:

* Choose 2 distinct names from ideas, call them ideaA and ideaB.
* Swap the first letters of ideaA and ideaB with each other.
* If both of the new names are not found in the original ideas, then the name ideaA ideaB (the concatenation of ideaA and ideaB, separated by a space) is a valid company name.
* Otherwise, it is not a valid name.

Return the number of distinct valid names for the company.



Example 1:

	Input: ideas = ["coffee","donuts","time","toffee"]
	Output: 6
	Explanation: The following selections are valid:
	- ("coffee", "donuts"): The company name created is "doffee conuts".
	- ("donuts", "coffee"): The company name created is "conuts doffee".
	- ("donuts", "time"): The company name created is "tonuts dime".
	- ("donuts", "toffee"): The company name created is "tonuts doffee".
	- ("time", "donuts"): The company name created is "dime tonuts".
	- ("toffee", "donuts"): The company name created is "doffee tonuts".
	Therefore, there are a total of 6 distinct company names.
	
	The following are some examples of invalid selections:
	- ("coffee", "time"): The name "toffee" formed after swapping already exists in the original array.
	- ("time", "toffee"): Both names are still the same after swapping and exist in the original array.
	- ("coffee", "toffee"): Both names formed after swapping already exist in the original array.

	
Example 2:

	Input: ideas = ["lack","back"]
	Output: 0
	Explanation: There are no valid selections. Therefore, 0 is returned.






Note:

	2 <= ideas.length <= 5 * 10^4
	1 <= ideas[i].length <= 10
	ideas[i] consists of lowercase English letters.
	All the strings in ideas are unique.


### 解析


根据题意， 给定一个字符串数组 ideas ，代表用于给公司命名的单词列表。 公司命名流程如下：

* 从想法中选择 2 个不同的名称，分别称为 ideaA 和 ideaB。
* 交换 ideaA 和 ideaB 的首字母。
* 如果在原始创意中都没有找到这两个新名称，则名称 ideaA ideaB 是有效的公司名称。否则，它不是一个有效的名称。

返回公司的不同有效名称的数量。

这道题考查的就是一个集合的使用，我们要想知道 ideas 组成合法的公司名的数量，只需要关注两个首字母及其后面的部分即可，定义一个字典 d 中，里面存放的对象是集合。我们首先遍历 ideas ，将相同首字母的从第一个字符开始的子串都放到一个集合里，然后我们比较这个字典中的当前键对应的集合 v1  和其他的键对应的集合 v2 所能形成的交集 s ，那么这两个键所能形成的有效公司名即为  (len(v1)-len(s)) * (len(v2)-len(s)) * 2 ，将其加到 result ，再去进行其他两个不同键的比较，最后返回 result 即为答案。

时间复杂度为 O(N) ，空间复杂度为 O(N) 。

### 解答
				
	class Solution(object):
	    def distinctNames(self, ideas):
	        """
	        :type ideas: List[str]
	        :rtype: int
	        """
	        d = collections.defaultdict(set)
	        for idea in ideas:
	            d[idea[0]].add(idea[1:])
	        result = 0
	        keys = list(d.keys())
	        N = len(keys)
	        for i in range(N):
	            for j in range(i+1, N):
	                k1,k2 = keys[i],keys[j]
	                v1,v2 = d[k1],d[k2]
	                s = v1 & v2
	                if k1 != k2:
	                    result += (len(v1)-len(s)) * (len(v2)-len(s)) * 2
	        return result

            	      
			
### 运行结果

	
	Runtime: 544 ms, faster than 91.67% of Python online submissions for Naming a Company.
	Memory Usage: 33 MB, less than 47.92% of Python online submissions for Naming a Company.

### 原题链接


https://leetcode.com/contest/weekly-contest-297/problems/naming-a-company/


您的支持是我最大的动力
