leetcode  401. Binary Watch（python）

### 描述

A binary watch has 4 LEDs on the top which represent the hours (0-11), and the 6 LEDs on the bottom represent the minutes (0-59). Each LED represents a zero or one, with the least significant bit on the right.

For example, the below binary watch reads "4:51".

![avatar](https://assets.leetcode.com/uploads/2021/04/08/binarywatch.jpg)

Given an integer turnedOn which represents the number of LEDs that are currently on, return all possible times the watch could represent. You may return the answer in any order.

The hour must not contain a leading zero.

	For example, "01:00" is not valid. It should be "1:00".
The minute must be consist of two digits and may contain a leading zero.

	For example, "10:2" is not valid. It should be "10:02".

Example 1:

	Input: turnedOn = 1
	Output: ["0:01","0:02","0:04","0:08","0:16","0:32","1:00","2:00","4:00","8:00"]
	
Example 2:

	Input: turnedOn = 9
	Output: []



Note:

	0 <= turnedOn <= 10

### 解析

根据题意，这里就是在找 4 块表示小时的 LED 灯和 6 块表示分钟的 LED 灯的所有组合的可能，我这里有点取巧，提前将各种可能出现的情况都列了出来，然后只需要将不同的情况拼接成时间字符串即可。

### 解答
				
	class Solution(object):
	    def readBinaryWatch(self, num):
	        """
	        :type num: int
	        :rtype: List[str]
	        """
	        r = []
	
	        def findH(n):
	            if n == 0:
	                return ['0']
	            elif n == 1:
	                return ['1', '2', '4', '8']
	            elif n == 2:
	                return ['3', '5', '9', '6', '10']
	            elif n == 3:
	                return ['7', '11']
	
	        def findM(n):
	            if n == 0:
	                return ['00']
	            elif n == 1:
	                return ['01', '02', '04', '08', '16', '32']
	            elif n == 2:
	                return ['03', '05', '09', '17', '33', '06', '10', '18', '34', '12', '20', '36', '24', '40', '48']
	            elif n == 3:
	                return ['07', '11', '19', '35', '13', '21', '37', '25', '41', '49', '14', '22', '38', '26', '42', '50',
	                        '28', '44', '52', '56']
	            elif n == 4:
	                return ['15', '23', '39', '27', '43', '51', '29', '45', '53', '57', '30', '46', '54', '58']
	            elif n == 5:
	                return ['31', '47', '55', '59']
	
	        for H in range(0, num + 1):
	            if H > 3:
	                continue
	            M = num - H
	            if M > 5:
	                continue
	            HS = findH(H)
	            MS = findM(M)
	            for h in HS:
	                for m in MS:
	                    r.append(h + ":" + m)
	        return r
	
	

            	      
			
### 运行结果

	Runtime: 12 ms, faster than 97.73% of Python online submissions for Binary Watch.
	Memory Usage: 13.6 MB, less than 15.91% of Python online submissions for Binary Watch.

原题链接：https://leetcode.com/problems/binary-watch/


您的支持是我最大的动力
