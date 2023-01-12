leetcode 39. Combination Sum （python）




### 描述



Given an array of distinct integers candidates and a target integer target, return a list of all unique combinations of candidates where the chosen numbers sum to target. You may return the combinations in any order.

The same number may be chosen from candidates an unlimited number of times. Two combinations are unique if the frequency of at least one of the chosen numbers is different.

It is guaranteed that the number of unique combinations that sum up to target is less than 150 combinations for the given input.

Example 1:

	Input: candidates = [2,3,6,7], target = 7
	Output: [[2,2,3],[7]]
	Explanation:
	2 and 3 are candidates, and 2 + 2 + 3 = 7. Note that 2 can be used multiple times.
	7 is a candidate, and 7 = 7.
	These are the only two combinations.



Note:


	1 <= candidates.length <= 30
	1 <= candidates[i] <= 200
	All elements of candidates are distinct.
	1 <= target <= 500

### 解析

根据题意，给定一个由不同整数组成的数组 candidates 和一个目标整数 target ，返回一个包含所有由 candidates 中元素所组成的和为 target 的列表组合，可以按任何顺序返回组合，并且可以从 candidates 中无限次选择相同的数字，只要数字的出现次数不同就可以作为新的组合。

这道题考查的其实就是 DFS ，使用递归解题会方便很多，为了减少遍历的次数，我们一开始将 candidates 进行了升序排序，这样如果某个递归阶段组合和已经大于 target ，之后的数字我们就可以直接跳过不去判断，节省时间。我们选取 candidates 中可能选取的元素并累加当前组合的 total ，直到 total 为 target 的时候，我们将这个过程中得到的组合加入结果列表 result 中，递归结束返回 result 即可。

为了更加形象的描述这个过程，我以例子一举例，用 2 表示某个可能组合的开头（另外还有 3 、6 、7 没有画，自己可以在纸上画一下，类似操作），来展示 DFS 递归过程，可以看到这个组合会尽可能地向下去找和为 target 的组合，被我划掉的就是当某个递归阶段的组合和大于 target 的时候，后面的递归就可以不用进行了，因为没有意义，最后剩下的有效的组合只有一种 [2,2,3] 。


### 解答
				

	class Solution(object):
	    def combinationSum(self, candidates, target):
	        result = []
	        candidates.sort()
	        def dfs(total, nums, index):
	            if total > target: return
	            if total == target:
	                if nums not in result:
	                    result.append(nums)
	                return
	            for i in range(index, len(candidates)):
	                dfs(total+candidates[i], nums+[candidates[i]], i)
	        dfs(0, [], 0)
	        return result
	        
            	      
			
### 运行结果


	Runtime: 120 ms, faster than 35.27% of Python online submissions for Combination Sum.
	Memory Usage: 13.9 MB, less than 6.67% of Python online submissions for Combination Sum.



### 原题链接

https://leetcode.com/problems/combination-sum/submissions/


您的支持是我最大的动力
