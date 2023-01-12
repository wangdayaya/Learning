
### 描述


Given an integer numRows, return the first numRows of Pascal's triangle.

In Pascal's triangle, each number is the sum of the two numbers directly above it as shown:




Example 1:

	
	Input: numRows = 5
	Output: [[1],[1,1],[1,2,1],[1,3,3,1],[1,4,6,4,1]]
	
Example 2:

	Input: numRows = 1
	Output: [[1]]





Note:


	1 <= numRows <= 30

### 解析

根据题意，给出来一个正整数 numRows ，要求返回 Pascal's Triangle 的前 numRows 行。

其实但凡上过素质教育的都知道这是杨辉三角，只不过国外人坐井观天，帕斯卡发现了这个规律，还怕别人抢它的功劳，按了一个自己的名字，后来就成了帕斯卡三角，其实比杨辉晚了近 400 年，洋人的格局可见一斑。

不过命名这个事这么看也挺重要的，如果命名成杨辉三角，大家肯定都知道这是中国人发明的，但是明明成帕斯卡三角，很多人就开始心里打鼓了，不确定是不是中国人发明的，所以掌握命名权是多么的重要，从这里再延伸一下，像各种节日申遗我们必须要争取，否则过个十几二十年，小孩子都以为中秋节是韩国的呢！


杨辉三角的规律题目中也给出来了，就是每个数字是其正上方的两个数字的总和，如图所示：

![](https://gimg2.baidu.com/image_search/src=http%3A%2F%2F5b0988e595225.cdn.sohucs.com%2Fimages%2F20200410%2Fceb23eeae6c44e67bd9321d801792c98.gif&refer=http%3A%2F%2F5b0988e595225.cdn.sohucs.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1637720616&t=be0244f74d9e515015ee85264eb321db)


解题思路也是按照这个来的，第一行是 [1] ，第二行是 [1,1]，从第三行开始除了第一个元素和最后一个元素，中间的元素都是这个规律。

### 解答
				

	class Solution(object):
	    def generate(self, numRows):
	        """
	        :type numRows: int
	        :rtype: List[List[int]]
	        """
	        result = []
	        for i in range(1, numRows+1):
	            tmp = [1]
	            for j in range(1, i-1):
	                tmp.append(result[-1][j-1]+ result[-1][j])
	            if i>=2:
	                tmp.append(1)
	            result.append(tmp)
	        return result
	                
	                
	            
            	      
			
### 运行结果

	Runtime: 50 ms, faster than 5.64% of Python online submissions for Pascal's Triangle.
	Memory Usage: 13.5 MB, less than 37.26% of Python online submissions for Pascal's Triangle.


原题链接：https://leetcode.com/problems/pascals-triangle/


您的支持是我最大的动力
