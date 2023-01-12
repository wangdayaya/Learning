leetcode 2383. Minimum Hours of Training to Win a Competition （python）




### 描述

You are entering a competition, and are given two positive integers initialEnergy and initialExperience denoting your initial energy and initial experience respectively. You are also given two 0-indexed integer arrays energy and experience, both of length n. You will face n opponents in order. The energy and experience of the i<sub>th</sub> opponent is denoted by energy[i] and experience[i] respectively. When you face an opponent, you need to have both strictly greater experience and energy to defeat them and move to the next opponent if available.

Defeating the i<sub>th</sub> opponent increases your experience by experience[i], but decreases your energy by energy[i]. Before starting the competition, you can train for some number of hours. After each hour of training, you can either choose to increase your initial experience by one, or increase your initial energy by one. Return the minimum number of training hours required to defeat all n opponents.





Example 1:

	Input: initialEnergy = 5, initialExperience = 3, energy = [1,4,3,2], experience = [2,6,3,1]
	Output: 8
	Explanation: You can increase your energy to 11 after 6 hours of training, and your experience to 5 after 2 hours of training.
	You face the opponents in the following order:
	- You have more energy and experience than the 0th opponent so you win.
	  Your energy becomes 11 - 1 = 10, and your experience becomes 5 + 2 = 7.
	- You have more energy and experience than the 1st opponent so you win.
	  Your energy becomes 10 - 4 = 6, and your experience becomes 7 + 6 = 13.
	- You have more energy and experience than the 2nd opponent so you win.
	  Your energy becomes 6 - 3 = 3, and your experience becomes 13 + 3 = 16.
	- You have more energy and experience than the 3rd opponent so you win.
	  Your energy becomes 3 - 2 = 1, and your experience becomes 16 + 1 = 17.
	You did a total of 6 + 2 = 8 hours of training before the competition, so we return 8.
	It can be proven that no smaller answer exists.

	
Example 2:

	Input: initialEnergy = 2, initialExperience = 4, energy = [1], experience = [3]
	Output: 0
	Explanation: You do not need any additional energy or experience to win the competition, so we return 0.




Note:


	n == energy.length == experience.length
	1 <= n <= 100	
	1 <= initialEnergy, initialExperience, energy[i], experience[i] <= 100

### 解析

根据题意，我们正在参加比赛，并给定两个正整数 initialEnergy 和 initialExperience，分别表示初始能量和初始体验。 还给定两个长度为 n 的 0 索引整数数组 energy 和 experience 。 然后依次面对 n 个对手。 第 i<sub>th</sub> 个对手的能量和经验分别用 energy[i] 和 experience[i] 表示。 当面对一个对手时，需要拥有足够大的经验和能量来击败他然后对阵下一个对手。

击败第 i<sub>th</sub> 个对手会增加 experience[i] 的经验，但会减少 energy[i] 的能量。 在开始比赛之前，我们可以训练几个小时。 每一小时的训练后，可以选择将初始经验加一，也可以选择初始能量加一。 返回击败所有 n 个对手所需的最少训练小时数。

解决这道题目只需要模拟题意写代码即可，这道题的题目有些绕，但是关键点就在于对阵对手的时候经验和能力都要比他高才行，所以我们在对阵每一名选手的时候我们要定义两个临时变量  increase_energy 和 increase_experience  ，如果发现当前的能量或者经验不比对手高，就要加上刚好超过对手的量才行，这些量都要计入结果 result 中，然后将当前的能量或者经验加上经过训练增加的量再与对手进行对阵即可，不断重复这个过程，最后返回 result 就是我们需要经过锻炼增加的最少的量。

时间复杂度为 O(N) ，空间复杂度为 O(1) 。

### 解答

	class Solution(object):
	    def minNumberOfHours(self, initialEnergy, initialExperience, energy, experience):
	        """
	        :type initialEnergy: int
	        :type initialExperience: int
	        :type energy: List[int]
	        :type experience: List[int]
	        :rtype: int
	        """
	        result = 0
	        N = len(energy)
	        for i in range(N):
	            increase_energy = 0
	            increase_experience = 0
	            if initialEnergy <= energy[i]:
	                increase_energy = energy[i] - initialEnergy + 1
	                result += increase_energy
	            if initialExperience <= experience[i]:
	                increase_experience = experience[i] - initialExperience + 1
	                result += increase_experience
	            initialEnergy = initialEnergy + increase_energy - energy[i]
	            initialExperience = initialExperience + increase_experience + experience[i]
	        return result



### 运行结果

	111 / 111 test cases passed.
	Status: Accepted
	Runtime: 51 ms
	Memory Usage: 13.2 MB
	Submitted: 0 minutes ago


### 原题链接

	https://leetcode.com/contest/weekly-contest-307/problems/minimum-hours-of-training-to-win-a-competition/


您的支持是我最大的动力
