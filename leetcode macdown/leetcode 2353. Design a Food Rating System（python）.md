leetcode 2353. Design a Food Rating System （python）




### 描述

Design a food rating system that can do the following:

* Modify the rating of a food item listed in the system.
* Return the highest-rated food item for a type of cuisine in the system.

Implement the FoodRatings class:

* FoodRatings(String[] foods, String[] cuisines, int[] ratings) Initializes the system. The food items are described by foods, cuisines and ratings, all of which have a length of n.
* foods[i] is the name of the ith food,
* cuisines[i] is the type of cuisine of the ith food, and
* ratings[i] is the initial rating of the ith food.
* void changeRating(String food, int newRating) Changes the rating of the food item with the name food.
* String highestRated(String cuisine) Returns the name of the food item that has the highest rating for the given type of cuisine. If there is a tie, return the item with the lexicographically smaller name.

Note that a string x is lexicographically smaller than string y if x comes before y in dictionary order, that is, either x is a prefix of y, or if i is the first position such that x[i] != y[i], then x[i] comes before y[i] in alphabetic order.



Example 1:


	Input
	["FoodRatings", "highestRated", "highestRated", "changeRating", "highestRated", "changeRating", "highestRated"]
	[[["kimchi", "miso", "sushi", "moussaka", "ramen", "bulgogi"], ["korean", "japanese", "japanese", "greek", "japanese", "korean"], [9, 12, 8, 15, 14, 7]], ["korean"], ["japanese"], ["sushi", 16], ["japanese"], ["ramen", 16], ["japanese"]]
	Output
	[null, "kimchi", "ramen", null, "sushi", null, "ramen"]
	
	Explanation
	FoodRatings foodRatings = new FoodRatings(["kimchi", "miso", "sushi", "moussaka", "ramen", "bulgogi"], ["korean", "japanese", "japanese", "greek", "japanese", "korean"], [9, 12, 8, 15, 14, 7]);
	foodRatings.highestRated("korean"); // return "kimchi"
	                                    // "kimchi" is the highest rated korean food with a rating of 9.
	foodRatings.highestRated("japanese"); // return "ramen"
	                                      // "ramen" is the highest rated japanese food with a rating of 14.
	foodRatings.changeRating("sushi", 16); // "sushi" now has a rating of 16.
	foodRatings.highestRated("japanese"); // return "sushi"
	                                      // "sushi" is the highest rated japanese food with a rating of 16.
	foodRatings.changeRating("ramen", 16); // "ramen" now has a rating of 16.
	foodRatings.highestRated("japanese"); // return "ramen"
	                                      // Both "sushi" and "ramen" have a rating of 16.
	                                      // However, "ramen" is lexicographically smaller than "sushi".
	




Note:

	1 <= n <= 2 * 10^4
	n == foods.length == cuisines.length == ratings.length
	1 <= foods[i].length, cuisines[i].length <= 10
	foods[i], cuisines[i] consist of lowercase English letters.
	1 <= ratings[i] <= 10^8
	All the strings in foods are distinct.
	food will be the name of a food item in the system across all calls to changeRating.
	cuisine will be a type of cuisine of at least one food item in the system across all calls to highestRated.
	At most 2 * 10^4 calls in total will be made to changeRating and highestRated.


### 解析

根据题意，设计一个可以执行以下操作的食品评级系统：

* 修改系统中列出的食物项目的评级。
* 返回系统中某类菜系评分最高的菜品。

实现 FoodRatings 类：

* FoodRatings(String[] foods, String[] foods, int[] rating) 初始化系统。食物项目由 foods 、cuisines 和 ratings 来描述，它们的长度都是 n。 foods[i] 是第 i 个食物的名称， cuisines[i] 是第 i 种食物的美食类型，ratings[i] 是第 i 个食物的初始评分。
* void changeRating(String food, int newRating) 更改名称为 food 的食物的评级。
* String highestRated(String cuisine) 返回对给定类型的美食具有最高评级的食物的名称。如果有相同的菜名，则返回具有按字典顺序较小的名称的项目。

先分析 highestRated 函数，它的主要功能是返回 cuisine 菜系中评分最高同时字典顺序最小的菜名，我们为了便于取菜系对应的菜，肯定要用到字典，但是菜系对应的菜要根据分数和名字排序，我们可以借用 SortedSet 的数据结构来解决，我们定义为 c2f ，这样就可以直接调用结果值，时间复杂度为 O(1) 。

分析 changeRating 函数，它的主要功能是对已经存在的菜的评分进行修改，这里有个需要注意的地方是，我们要将已经记录在 c2f 中的旧的分数及其菜名去掉，加入新的分数及其菜名，同时更新 f2rc 中菜名对应的新评分和菜系。时间复杂度主要消耗在 c2f 中的 SortedSet 的排序上，时间复杂度为 O(NlogN) 。

整个解法的时间复杂度为 O(NlogN) ，空间复杂度为 O(N) 。
### 解答

	from sortedcontainers import SortedSet
	class FoodRatings(object):
	    def __init__(self, foods, cuisines, ratings):
	        """
	        :type foods: List[str]
	        :type cuisines: List[str]
	        :type ratings: List[int]
	        """
	        self.f2rc = collections.defaultdict(tuple)
	        self.c2f = collections.defaultdict(SortedSet)
	        for f, c, r in zip(foods, cuisines, ratings):
	            self.f2rc[f] = (r, c)
	            self.c2f[c].add((-r, f))
	
	    def changeRating(self, food, newRating):
	        """
	        :type food: str
	        :type newRating: int
	        :rtype: None
	        """
	        oldRaing, cuisine = self.f2rc[food]
	        self.c2f[cuisine].remove((-oldRaing, food))
	        self.c2f[cuisine].add((-newRating, food))
	        self.f2rc[food] = (newRating, cuisine)
	
	    def highestRated(self, cuisine):
	        """
	        :type cuisine: str
	        :rtype: str
	        """
	        return self.c2f[cuisine][0][1]

### 运行结果

	76 / 76 test cases passed.
	Status: Accepted
	Runtime: 1805 ms
	Memory Usage: 53.2 MB

### 原题链接

https://leetcode.com/contest/weekly-contest-303/problems/design-a-food-rating-system/


您的支持是我最大的动力
