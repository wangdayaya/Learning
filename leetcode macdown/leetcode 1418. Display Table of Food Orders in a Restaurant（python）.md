leetcode 1418. Display Table of Food Orders in a Restaurant （python）

### 描述

Given the array orders, which represents the orders that customers have done in a restaurant. More specifically orders[i]=[customerName<sub>i</sub>,tableNumber<sub>i</sub>,foodItem<sub>i</sub>] where customerName<sub>i</sub> is the name of the customer, tableNumber<sub>i</sub> is the table customer sit at, and foodItem<sub>i</sub> is the item customer orders.

Return the restaurant's “display table”. The “display table” is a table whose row entries denote how many of each food item each table ordered. The first column is the table number and the remaining columns correspond to each food item in alphabetical order. The first row should be a header whose first column is “Table”, followed by the names of the food items. Note that the customer names are not part of the table. Additionally, the rows should be sorted in numerically increasing order.



Example 1:

	Input: orders = [["David","3","Ceviche"],["Corina","10","Beef Burrito"],["David","3","Fried Chicken"],["Carla","5","Water"],["Carla","5","Ceviche"],["Rous","3","Ceviche"]]
	Output: [["Table","Beef Burrito","Ceviche","Fried Chicken","Water"],["3","0","2","1","0"],["5","0","1","0","1"],["10","1","0","0","0"]] 
	Explanation:
	The displaying table looks like:
	Table,Beef Burrito,Ceviche,Fried Chicken,Water
	3    ,0           ,2      ,1            ,0
	5    ,0           ,1      ,0            ,1
	10   ,1           ,0      ,0            ,0
	For the table 3: David orders "Ceviche" and "Fried Chicken", and Rous orders "Ceviche".
	For the table 5: Carla orders "Water" and "Ceviche".
	For the table 10: Corina orders "Beef Burrito". 

	
Example 2:

	
	Input: orders = [["James","12","Fried Chicken"],["Ratesh","12","Fried Chicken"],["Amadeus","12","Fried Chicken"],["Adam","1","Canadian Waffles"],["Brianna","1","Canadian Waffles"]]
	Output: [["Table","Canadian Waffles","Fried Chicken"],["1","2","0"],["12","0","3"]] 
	Explanation: 
	For the table 1: Adam and Brianna order "Canadian Waffles".
	For the table 12: James, Ratesh and Amadeus order "Fried Chicken".

Example 3:

	Input: orders = [["Laura","2","Bean Burrito"],["Jhon","2","Beef Burrito"],["Melissa","2","Soda"]]
	Output: [["Table","Bean Burrito","Beef Burrito","Soda"],["2","1","1","1"]]

	


Note:

* 1 <= orders.length <= 5 * 10^4
* orders[i].length == 3
* 1 <= customerNamei.length, foodItemi.length <= 20
* customerNamei and foodItemi consist of lowercase and uppercase English letters and the space character.
* tableNumber<sub>i</sub> is a valid integer between 1 and 500.


### 解析


根据题意，给定数组 orders，它表示客户在餐厅下的订单。 orders[i]=[customerName<sub>i</sub>,tableNumber<sub>i</sub>,foodItem<sub>i</sub>] 其中 customerName<sub>i</sub> 是客户的姓名，tableNumber<sub>i</sub> 是客户所在的餐桌，foodItem<sub>i</sub> 是客户订单的项目。

题目要求我们返回一个餐厅的“display table”。 “display table”是一个表，第一行是一个表头，其第一列是“ Table ”，其余列按字母顺序排列的每个食品项目。从第二行开始表示的是每张桌子订购的每种食品的数量。需要注意的是客户名称不是表的一部分。 此外应按桌号递增顺序对行进行排序。

看起来题目比较复杂，其实客户名没啥用，关键看桌号和菜名，看了例子一基本就能知道题目要求了，思路也比较简单：

* 初始化空列表 result 表示结果，初始化空列表 head 表示表头，初始化空字典 d 表示每个桌号对应的菜以及数量
* 遍历所有的订单 orders ，用 d 记录桌号及其对应的菜品和菜品数量，并且将菜名无重复放入 head 中
* 遍历结束将 head 按照字典序排列并且将 ‘Table’ 字符串插入到最前面形成最终的表头，并将 head 追加到 result 中
* 对字典 d 按照桌号升序进行排序，然后遍历 d 的键值对，按照 head[1:] 中菜的顺序，将每个桌号的菜及其数量都加入到新的列表 row 中，并将 row 追加到 result 中
* 遍历结束返回 result 即可


### 解答
				

	class Solution(object):
	    def displayTable(self, orders):
	        """
	        :type orders: List[List[str]]
	        :rtype: List[List[str]]
	        """
	        result = []
	        head = []
	        d = {}
	        for order in orders:
	            if order[1] not in d:
	                d[order[1]] = {}
	            if order[2] not in d[order[1]]:
	                d[order[1]][order[2]] = 1
	            else:
	                d[order[1]][order[2]] += 1
	            if order[2] not in head:
	                head.append(order[2])
	        head.sort()
	        head = ['Table'] + head
	        result.append(head)
	        d = sorted(d.items(), key=lambda d: int(d[0]))
	        for k, v in d:
	            row = [k]
	            for food in head[1:]:
	                if food not in v:
	                    row.append('0')
	                else:
	                    row.append(str(v[food]))
	            result.append(row)
	        return result
	        
            	      
			
### 运行结果

	Runtime: 400 ms, faster than 94.12% of Python online submissions for Display Table of Food Orders in a Restaurant.
	Memory Usage: 22.9 MB, less than 47.06% of Python online submissions for Display Table of Food Orders in a Restaurant.


原题链接：https://leetcode.com/problems/display-table-of-food-orders-in-a-restaurant/



您的支持是我最大的动力
