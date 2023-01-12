leetcode  455. Assign Cookies（python）

### 描述


Assume you are an awesome parent and want to give your children some cookies. But, you should give each child at most one cookie.

Each child i has a greed factor g[i], which is the minimum size of a cookie that the child will be content with; and each cookie j has a size s[j]. If s[j] >= g[i], we can assign the cookie j to the child i, and the child i will be content. Your goal is to maximize the number of your content children and output the maximum number.




Example 1:

	Input: g = [1,2,3], s = [1,1]
	Output: 1
	Explanation: You have 3 children and 2 cookies. The greed factors of 3 children are 1, 2, 3. 
	And even though you have 2 cookies, since their size is both 1, you could only make the child whose greed factor is 1 content.
	You need to output 1.

	
Example 2:

	Input: g = [1,2], s = [1,2,3]
	Output: 2
	Explanation: You have 2 children and 3 cookies. The greed factors of 2 children are 1, 2. 
	You have 3 cookies and their sizes are big enough to gratify all of the children, 
	You need to output 2.






Note:

	1 <= g.length <= 3 * 10^4
	0 <= s.length <= 3 * 10^4
	1 <= g[i], s[j] <= 2^31 - 1


### 解析

根据题意，就是有蛋糕尺寸列表 s ，孩子得到满足的最小蛋糕尺寸列表 g ，只能给每个孩子最多一个蛋糕，并且要达到其满足的最小蛋糕尺寸才行，找出得到满足的孩子有几个。先将 g 和 s 进行排序，然后遍历 g 和 s ，如果某蛋糕满足孩子的蛋糕尺寸要求，则将该蛋糕去掉，并且结果加 1 ，遍历结束即可得到答案。


### 解答
				

	class Solution(object):
	    def findContentChildren(self, g, s):
	        """
	        :type g: List[int]
	        :type s: List[int]
	        :rtype: int
	        """
	        r = 0
	        g.sort()
	        s.sort()
	        for i,v in enumerate(g):
	            for j,c in enumerate(s):
	                if v<=c:
	                    s.pop(j)
	                    r+=1
	                    break
	        return r
            	      
			
### 运行结果

	Runtime: 1088 ms, faster than 10.12% of Python online submissions for Assign Cookies.
	Memory Usage: 14.7 MB, less than 94.16% of Python online submissions for Assign Cookies.


### 解析

上面的双重循环太耗时间，可以只用一重循环找到答案，定义 i 表示 g 的索引，res 表示结果。现将 g 和 s 进行排序，然后遍历 s ，如果 g 为空表示没有孩子列表，则直接返回结果 0 。否则如果当前尺寸满足孩子要求时，结果加 1 ，i 也加 1 。继续循环上述操作，遍历结束会得到结果。


### 解答
				
	class Solution(object):
	    def findContentChildren(self, g, s):
	        """
	        :type g: List[int]
	        :type s: List[int]
	        :rtype: int
	        """
	        g.sort()
	        s.sort()
	        res = 0
	        i = 0
	        for e in s:
	            if i == len(g):
	                break
	            if e >= g[i]:
	                res += 1
	                i += 1
	        return res
			
### 运行结果

	Runtime: 124 ms, faster than 98.83% of Python online submissions for Assign Cookies.
	Memory Usage: 14.9 MB, less than 54.09% of Python online submissions for Assign Cookies.

原题链接：https://leetcode.com/problems/assign-cookies/



您的支持是我最大的动力
