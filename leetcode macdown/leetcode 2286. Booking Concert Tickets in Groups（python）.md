leetcode 2286. Booking Concert Tickets in Groups （python）

### 每日经典

《》 ——（）


### 描述


A concert hall has n rows numbered from 0 to n - 1, each with m seats, numbered from 0 to m - 1. You need to design a ticketing system that can allocate seats in the following cases:

* If a group of k spectators can sit together in a row.
* If every member of a group of k spectators can get a seat. They may or may not sit together.

Note that the spectators are very picky. Hence:

* They will book seats only if each member of their group can get a seat with row number less than or equal to maxRow. maxRow can vary from group to group.
* In case there are multiple rows to choose from, the row with the smallest number is chosen. If there are multiple seats to choose in the same row, the seat with the smallest number is chosen.

Implement the BookMyShow class:

* BookMyShow(int n, int m) Initializes the object with n as number of rows and m as number of seats per row.
* int[] gather(int k, int maxRow) Returns an array of length 2 denoting the row and seat number (respectively) of the first seat being allocated to the k members of the group, who must sit together. In other words, it returns the smallest possible r and c such that all [c, c + k - 1] seats are valid and empty in row r, and r <= maxRow. Returns [] in case it is not possible to allocate seats to the group.
* boolean scatter(int k, int maxRow) Returns true if all k members of the group can be allocated seats in rows 0 to maxRow, who may or may not sit together. If the seats can be allocated, it allocates k seats to the group with the smallest row numbers, and the smallest possible seat numbers in each row. Otherwise, returns false.



Example 1:

	Input
	["BookMyShow", "gather", "gather", "scatter", "scatter"]
	[[2, 5], [4, 0], [2, 0], [5, 1], [5, 1]]
	Output
	[null, [0, 0], [], true, false]
	
	Explanation
	BookMyShow bms = new BookMyShow(2, 5); // There are 2 rows with 5 seats each 
	bms.gather(4, 0); // return [0, 0]
	                  // The group books seats [0, 3] of row 0. 
	bms.gather(2, 0); // return []
	                  // There is only 1 seat left in row 0,
	                  // so it is not possible to book 2 consecutive seats. 
	bms.scatter(5, 1); // return True
	                   // The group books seat 4 of row 0 and seats [0, 3] of row 1. 
	bms.scatter(5, 1); // return False
	                   // There are only 2 seats left in the hall.

	





Note:

	1 <= n <= 5 * 10^4
	1 <= m, k <= 10^9
	0 <= maxRow <= n - 1
	At most 5 * 10^4 calls in total will be made to gather and scatter.


### 解析


根据题意，这道题就是让我们给一组顾客预定座位，需要我们设计一个函数来实现具体的需求：

* BookMyShow(int n, int m) 主要是初始化 n 行 m 列个空座位
* int[] gather(int k, int maxRow) 主要是判断这 k 个人能不能挨着一起坐到前 maxRow 杭的某一行
* boolean scatter(int k, int maxRow) 主要判断的是这 k 个人能不能分散坐到前  maxRow 行中



### 解答
				


            	      
			
### 运行结果




### 原题链接

https://leetcode.com/contest/biweekly-contest-79/problems/booking-concert-tickets-in-groups/


您的支持是我最大的动力
