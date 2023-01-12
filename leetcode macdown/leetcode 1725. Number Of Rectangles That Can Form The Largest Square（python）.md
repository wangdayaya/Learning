leetcode  1725. Number Of Rectangles That Can Form The Largest Square（python）｜Python 主题月



本文正在参加「Python主题月」，详情查看 [活动链接](https://juejin.cn/post/6979532761954533390/)
### 描述


You are given an array rectangles where rectangles[i] = [li, wi] represents the ith rectangle of length li and width wi.

You can cut the ith rectangle to form a square with a side length of k if both k <= li and k <= wi. For example, if you have a rectangle [4,6], you can cut it to get a square with a side length of at most 4.

Let maxLen be the side length of the largest square you can obtain from any of the given rectangles.

Return the number of rectangles that can make a square with a side length of maxLen.

 


Example 1:


	Input: rectangles = [[5,8],[3,9],[5,12],[16,5]]
	Output: 3
	Explanation: The largest squares you can get from each rectangle are of lengths [5,3,5,5].
	The largest possible square is of length 5, and you can get it out of 3 rectangles.
	
Example 2:

	Input: rectangles = [[2,3],[3,7],[4,3],[3,7]]
	Output: 3





Note:

	1 <= rectangles.length <= 1000
	rectangles[i].length == 2
	1 <= li, wi <= 109
	li != wi


### 解析


根据题意，就是给出了矩形列表，将每个矩形切割可以形成正方形，问可以形成的边长最大的正方形的个数。正方形的边长就是靠矩形的较小边，所以找出所有矩形的最小边，然后统计个数字典，找出最大边的出现个数即可。这里用到了 Python 的内置函数
collections.Counter() ，有点取巧。

### 解答
				

	class Solution(object):
	    def countGoodRectangles(self, rectangles):
	        """
	        :type rectangles: List[List[int]]
	        :rtype: int
	        """
	        c = collections.Counter([min(r) for r in rectangles])
	        k = c.keys()
	        return c[max(k)]
            	      
			
### 运行结果

	Runtime: 160 ms, faster than 18.45% of Python online submissions for Number Of Rectangles That Can Form The Largest Square.
	Memory Usage: 14 MB, less than 58.67% of Python online submissions for Number Of Rectangles That Can Form The Largest Square.



## 解析

上面的直接用到了内置函数，如果不用内置函数，直接使用字典 d 保存每个矩形中的较小的边的个数，最后找到最大的边的个数即可。这里结果比较令人满意，竟然双双接近 100%，其实 leetcode 这个测评我觉得和网络环境有关系，有时候特别快，有时候很一般。

## 解答

	class Solution(object):
	    def countGoodRectangles(self, rectangles):
	        """
	        :type rectangles: List[List[int]]
	        :rtype: int
	        """
	        d = {}
	        for r in rectangles:
	            p = min(r)  
	            if p in d:
	                d[p] += 1 
	            else:
	                d[p] = 1 
	        return d[max(d.keys())]  

## 运行结果

	Runtime: 140 ms, faster than 97.60% of Python online submissions for Number Of Rectangles That Can Form The Largest Square.
	Memory Usage: 13.9 MB, less than 99.04% of Python online submissions for Number Of Rectangles That Can Form The Largest Square.

原题链接：https://leetcode.com/problems/number-of-rectangles-that-can-form-the-largest-square/



您的支持是我最大的动力
