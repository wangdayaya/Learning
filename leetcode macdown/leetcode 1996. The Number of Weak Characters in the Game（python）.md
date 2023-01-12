


### 描述

You are playing a game that contains multiple characters, and each of the characters has two main properties: attack and defense. You are given a 2D integer array properties where properties[i] = [attack<sub>i</sub>, defense<sub>i</sub>] represents the properties of the i<sub>th</sub> character in the game. A character is said to be weak if any other character has both attack and defense levels strictly greater than this character's attack and defense levels. More formally, a character i is said to be weak if there exists another character j where attack<sub>j</sub> > attack<sub>i</sub> and defense<sub>j</sub> > defense<sub>i</sub>.

Return the number of weak characters.



Example 1:

	Input: properties = [[5,5],[6,3],[3,6]]
	Output: 0
	Explanation: No character has strictly greater attack and defense than the other.

	
Example 2:


	Input: properties = [[2,2],[3,3]]
	Output: 1
	Explanation: The first character is weak because the second character has a strictly greater attack and defense.





Note:


	2 <= properties.length <= 10^5
	properties[i].length == 2
	1 <= attack<sub>i</sub>, defence<sub>i</sub> <= 10^5

### 解析

根据题意，我们正在玩一个包含多个角色的游戏，每个角色都有两个主要属性：攻击和防御。 给定一个二维整数数组 properties ，其中 properties[i] = [attack<sub>i</sub>, defence<sub>i</sub>] 表示第 i<sub>th</sub> 个角色的属性。如果存在一个其他角色的攻击和防御等级都严格高于该角色的攻击和防御等级，则称该角色为弱角色。 更通俗的讲就是，如果存在另一个角色索引 j ，其中 attack<sub>j</sub> > attack<sub>i</sub> 和 defence <sub>j</sub> > defence <sub>i</sub>，则称索引为 i 得角色为弱角色。返回弱角色的数量。

这道题其实考察的就是排序，使用一次遍历的方法就能 AC ，我们知道题目要求的是弱角色的数量，所以只需要计数即可。我要判断的维度有攻击力和防御力两个维度，那么我们按照物理学中的“控制变量法”的思想，先判断一个维度，再判断另一个维度即可。要想实现“严格大于”的这个要求，肯定要先排序，我们按照攻击力降序、防御力升序的方式进行排序，这样我们在确定弱角色攻击力满足“严格小于”的时候，再去判断防御力“严格小于”，即可找到弱角色。

然后我们遍历 properties ，如果当前的角色的攻击力小于之前遍历过的角色的攻击力，我们并不能判断其就是弱角色，因为防御力不确定，所以为了方便我们要使用一个变量 max\_defense 来记录遍历过的角色中的最大防御力，这样如果当前角色的防御力也严格小于 max\_defense ，说明其肯定是若角色，结果 result 加一即可。
这里其实还要判断攻击力相同但是防御力严格小于之前角色的这种情况，但是我们这种写法可以省去这个判断步骤，因为我们按照防御力升序排序，这样如果当前防御力严格小于之前最大防御力的时候，攻击力一定不可能相等，因为攻击力相等的时候防御力是按照升序排序的。

按照上面的方式，一次遍历完得到的 result 就是最后的结果，返回即可。时间复杂度为 O(logN + N) ，N 是 properties 的长度，因为要先对 properties 进行排序然后再对其进行遍历，空间复杂度为 O(1) ，因为没有开辟新的空间。

### 解答

	class Solution(object):
	    def numberOfWeakCharacters(self, properties):
	        """
	        :type properties: List[List[int]]
	        :rtype: int
	        """
	        properties.sort(key=lambda x: (-x[0], x[1]))
	        result = max_defense = 0
	        for _, defense in properties:
	            if defense < max_defense:
	                result += 1
	            else:
	                max_defense = defense
	        return result
	


### 运行结果

	Runtime: 2301 ms, faster than 84.75% of Python online submissions for The Number of Weak Characters in the Game.
	Memory Usage: 69.6 MB, less than 42.37% of Python online submissions for The Number of Weak Characters in the Game.

### 原题链接

https://leetcode.com/problems/the-number-of-weak-characters-in-the-game/


您的支持是我最大的动力
