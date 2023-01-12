leetcode  2303. Calculate Amount Paid in Taxes（python）




### 描述


You are given a 0-indexed 2D integer array brackets where brackets[i] = [upperi, percenti] means that the ith tax bracket has an upper bound of upperi and is taxed at a rate of percenti. The brackets are sorted by upper bound (i.e. upperi-1 < upperi for 0 < i < brackets.length).

Tax is calculated as follows:

* The first upper0 dollars earned are taxed at a rate of percent0.
* The next upper1 - upper0 dollars earned are taxed at a rate of percent1.
* The next upper2 - upper1 dollars earned are taxed at a rate of percent2.
* And so on.

You are given an integer income representing the amount of money you earned. Return the amount of money that you have to pay in taxes. Answers within 10^-5 of the actual answer will be accepted.


Example 1:


	Input: brackets = [[3,50],[7,10],[12,25]], income = 10
	Output: 2.65000
	Explanation:
	The first 3 dollars you earn are taxed at 50%. You have to pay $3 * 50% = $1.50 dollars in taxes.
	The next 7 - 3 = 4 dollars you earn are taxed at 10%. You have to pay $4 * 10% = $0.40 dollars in taxes.
	The final 10 - 7 = 3 dollars you earn are taxed at 25%. You have to pay $3 * 25% = $0.75 dollars in taxes.
	You have to pay a total of $1.50 + $0.40 + $0.75 = $2.65 dollars in taxes.
	
Example 2:


	Input: brackets = [[1,0],[4,25],[5,50]], income = 2
	Output: 0.25000
	Explanation:
	The first dollar you earn is taxed at 0%. You have to pay $1 * 0% = $0 dollars in taxes.
	The second dollar you earn is taxed at 25%. You have to pay $1 * 25% = $0.25 dollars in taxes.
	You have to pay a total of $0 + $0.25 = $0.25 dollars in taxes.

Example 3:

	Input: brackets = [[2,50]], income = 0
	Output: 0.00000
	Explanation:
	You have no income to tax, so you have to pay a total of $0 dollars in taxes.

	



Note:

	1 <= brackets.length <= 100
	1 <= upperi <= 1000
	0 <= percenti <= 100
	0 <= income <= 1000
	upperi is sorted in ascending order.
	All the values of upperi are unique.
	The upper bound of the last tax bracket is greater than or equal to income.


### 解析


根据题意，给定一个 0 索引的 2D 整数数组 brackets ，其中 brackets[i] = [upper<sub>i</sub>, percent<sub>i</sub>] 表示第 i 个税级的上限为 upper<sub>i</sub>，并按百分比税率征税。 brackets 按上限排序。

税金计算如下：

* 赚取的 upper<sub>0</sub> 美元按 percent<sub>0</sub> 的税率征税。
* 赚取的下一个 upper1</sub> - upper<sub>0</sub> 美元按 percent<sub>1</sub> 征税。
* 接下来的 upper<sub>2</sub> - upper<sub>1</sub> 美元按 percent<sub>2</sub> 征税。
* 以此类推

给定一个整数 income 代表您赚取的金额。返回您必须缴纳的税款。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答
				

	class Solution(object):
	    def calculateTax(self, brackets, income):
	        """
	        :type brackets: List[List[int]]
	        :type income: int
	        :rtype: float
	        """
	        result = 0
	        pre = 0
	        for (b, p) in brackets:
	            if b <= income:
	                result += (b - pre) * p / float(100)
	            else:
	                result += (income - pre) * p / float(100)
	                break
	            pre = b
	        return result
            	      
			
### 运行结果


	225 / 225 test cases passed.
	Status: Accepted
	Runtime: 48 ms
	Memory Usage: 13.4 MB

### 原题链接


https://leetcode.com/contest/weekly-contest-297/problems/calculate-amount-paid-in-taxes/



您的支持是我最大的动力
