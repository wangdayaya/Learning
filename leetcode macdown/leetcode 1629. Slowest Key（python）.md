leetcode 1629. Slowest Key （python）

### 描述

A newly designed keypad was tested, where a tester pressed a sequence of n keys, one at a time.

You are given a string keysPressed of length n, where keysPressed[i] was the ith key pressed in the testing sequence, and a sorted list releaseTimes, where releaseTimes[i] was the time the ith key was released. Both arrays are 0-indexed. The 0th key was pressed at the time 0, and every subsequent key was pressed at the exact time the previous key was released.

The tester wants to know the key of the keypress that had the longest duration. The ith keypress had a duration of releaseTimes[i] - releaseTimes[i - 1], and the 0th keypress had a duration of releaseTimes[0].

Note that the same key could have been pressed multiple times during the test, and these multiple presses of the same key may not have had the same duration.

Return the key of the keypress that had the longest duration. If there are multiple such keypresses, return the lexicographically largest key of the keypresses.






Example 1:


	Input: releaseTimes = [9,29,49,50], keysPressed = "cbcd"
	Output: "c"
	Explanation: The keypresses were as follows:
	Keypress for 'c' had a duration of 9 (pressed at time 0 and released at time 9).
	Keypress for 'b' had a duration of 29 - 9 = 20 (pressed at time 9 right after the release of the previous character and released at time 29).
	Keypress for 'c' had a duration of 49 - 29 = 20 (pressed at time 29 right after the release of the previous character and released at time 49).
	Keypress for 'd' had a duration of 50 - 49 = 1 (pressed at time 49 right after the release of the previous character and released at time 50).
	The longest of these was the keypress for 'b' and the second keypress for 'c', both with duration 20.
	'c' is lexicographically larger than 'b', so the answer is 'c'.
	
Example 2:


	Input: releaseTimes = [12,23,36,46,62], keysPressed = "spuda"
	Output: "a"
	Explanation: The keypresses were as follows:
	Keypress for 's' had a duration of 12.
	Keypress for 'p' had a duration of 23 - 12 = 11.
	Keypress for 'u' had a duration of 36 - 23 = 13.
	Keypress for 'd' had a duration of 46 - 36 = 10.
	Keypress for 'a' had a duration of 62 - 46 = 16.
	The longest of these was the keypress for 'a' with duration 16.



Note:

	releaseTimes.length == n
	keysPressed.length == n
	2 <= n <= 1000
	1 <= releaseTimes[i] <= 10^9
	releaseTimes[i] < releaseTimes[i+1]
	keysPressed contains only lowercase English letters.


### 解析

根据题意，就是找出按下时间最长且字典序最大的键。可以初始化 result 为 keysPressed[0] 表示按下时间最长的键，初始化 time 为 releaseTimes[0] 表示最长的耗时时间。然后遍历 keysPressed ，当索引为 0 的时候直接进行下一个循环，因为已经将值附给了 result 和 time 了。如果某个键的按键时间大于 time 或者按键时间等于 time 且 result 的字典序小于某个键，则用该键的数据更新 result 和 time 。遍历结束得到的 result 即为结果。


### 解答
				

	class Solution(object):
	    def slowestKey(self, releaseTimes, keysPressed):
	        """
	        :type releaseTimes: List[int]
	        :type keysPressed: str
	        :rtype: str
	        """
	        result = keysPressed[0]
	        time = releaseTimes[0]
	        for i,key in enumerate(keysPressed):
	
	            if i == 0:
	                continue
	            elif releaseTimes[i]-releaseTimes[i-1]>time or (releaseTimes[i]-releaseTimes[i-1]==time and result<key):
	                result = key
	                time = releaseTimes[i]-releaseTimes[i-1]
	        return result
            	      
			
### 运行结果

	Runtime: 44 ms, faster than 48.38% of Python online submissions for Slowest Key.
	Memory Usage: 13.7 MB, less than 21.43% of Python online submissions for Slowest Key


原题链接：https://leetcode.com/problems/slowest-key/



您的支持是我最大的动力
