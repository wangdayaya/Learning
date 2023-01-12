leetcode 2007. Find Original Array From Doubled Array （python）




### 描述

An integer array original is transformed into a doubled array changed by appending twice the value of every element in original, and then randomly shuffling the resulting array. Given an array changed, return original if changed is a doubled array. If changed is not a doubled array, return an empty array. The elements in original may be returned in any order.



Example 1:


	Input: changed = [1,3,4,2,6,8]
	Output: [1,3,4]
	Explanation: One possible original array could be [1,3,4]:
	- Twice the value of 1 is 1 * 2 = 2.
	- Twice the value of 3 is 3 * 2 = 6.
	- Twice the value of 4 is 4 * 2 = 8.
	Other original arrays could be [4,3,1] or [3,1,4].
	
Example 2:


	Input: changed = [6,3,0,1]
	Output: []
	Explanation: changed is not a doubled array.

Example 3:


	Input: changed = [1]
	Output: []
	Explanation: changed is not a doubled array.


Note:

	1 <= changed.length <= 10^5
	0 <= changed[i] <= 10^5


### 解析

根据题意，一个整数数组 original 被转换为一个加倍数组 changed ，通过将原始数组中每个元素的两倍值追加到原始数组后面，然后随机打乱数组形成的加倍数组。给定一个数组 changed ，如果 changed 是一个加倍数组，则返回原始数组。 如果 changed 不是双倍数组，则返回一个空数组。 原始元素可以按任何顺序返回。

这道题要使用到排序和哈希来进行解题，因为加倍数组的定义原因，肯定有一半的数字比另一半的数字大，所以先将 changed 经过升序排序，这样便于我们进行后面的判断。然后我们再定义一个字典  doubles 来统计双倍数的出现的次数，然后开始遍历 changed ：

* 如果当前的 num 不是双倍数，那么我们就将其加入到结果 result 中，表示我们找到了一个原始数字，并且将 doubles 中 num*2 的个数加一，并且理论上会出现一个原始数字对应的双倍数
* 如果当前的 num 是双倍数，那么我们就将其在 doubles 中的个数减一，表示上面的理论上出现的数字在 num 中真的出现了

遍历结束之后，我们得到的 result 里面记录的就是可能的原始数组，doubles 中记录的就是可能的双倍数组，如果 result 的长度的 2 倍等于 changed ，说明 result 就是 changed 的原始数组，否则直接返回空数组。

N 为数组长度，时间复杂度为 O(NlogN + N) ，因为要经过前面的排序和后面的遍历，空间复杂度为 O(N) ，因为用到了字典记录。

### 解答

	class Solution(object):
	    def findOriginalArray(self, changed):
	        """
	        :type changed: List[int]
	        :rtype: List[int]
	        """
	        doubles = collections.Counter()
	        result = []
	        changed.sort()
	        for num in changed:
	            if doubles[num] == 0:
	                result.append(num)
	                doubles[num*2] += 1
	            else:
	                doubles[num] -= 1
	        if len(result) * 2 == len(changed):
	            return result
	        else:
	            return []

### 运行结果


	178 / 178 test cases passed.
	Status: Accepted
	Runtime: 2672 ms
	Memory Usage: 31.4 MB

### 原题链接

https://leetcode.com/problems/find-original-array-from-doubled-array/


您的支持是我最大的动力
