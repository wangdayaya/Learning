leetcode  806. Number of Lines To Write String（python）

### 描述


You are given a string s of lowercase English letters and an array widths denoting how many pixels wide each lowercase English letter is. Specifically, widths[0] is the width of 'a', widths[1] is the width of 'b', and so on.

You are trying to write s across several lines, where each line is no longer than 100 pixels. Starting at the beginning of s, write as many letters on the first line such that the total width does not exceed 100 pixels. Then, from where you stopped in s, continue writing as many letters as you can on the second line. Continue this process until you have written all of s.

Return an array result of length 2 where:

* result[0] is the total number of lines.
* result[1] is the width of the last line in pixels.



Example 1:

	Input: widths = [10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10], s = "abcdefghijklmnopqrstuvwxyz"
	Output: [3,60]
	Explanation: You can write s as follows:
	abcdefghij  // 100 pixels wide
	klmnopqrst  // 100 pixels wide
	uvwxyz      // 60 pixels wide
	There are a total of 3 lines, and the last line is 60 pixels wide.

	
Example 2:

	Input: widths = [4,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10,10], s = "bbbcccdddaaa"
	Output: [2,4]
	Explanation: You can write s as follows:
	bbbcccdddaa  // 98 pixels wide
	a            // 4 pixels wide
	There are a total of 2 lines, and the last line is 4 pixels wide.




Note:

	widths.length == 26
	2 <= widths[i] <= 10
	1 <= s.length <= 1000
	s contains only lowercase English letters.


### 解析

根据题意，widths 就是给出了 26 个小写英文字母中，每个字母所占的像素的大小，s 是给出了一个小写字母的字符串，定义每行最多有 100 个像素，将 s 中每个字母按照顺序从左到右排列，如果该行字母所占的像素和超出了 100 个像素，则从下一行开始排列字母，最后求一共占了多少行，并且最后一行的像素大小为多少。

* 初始化一个结果 result ，result[0] 表示共占了多少行，result[1] 表示最后一行的像素大小，定义最后一行的像素和为 last_row 
* 遍历 s 中的每个字符，每个字符所占的像素大小为 widths[ord(c)-97] ，如果 last\_row+w <= 100 表示该行像素还有空余，last_row 增加 w 个像素即可
* 否则表示已经超出了该行的像素极限，从下一行开始存放字符，所以 last_row 为 w ，并且一共占的行树增加一
* 遍历结束的时候 result[1] 为 last_row 
* 返回 result 即可


### 解答
				

	class Solution(object):
	    def numberOfLines(self, widths, s):
	        """
	        :type widths: List[int]
	        :type s: str
	        :rtype: List[int]
	        """
	        result = [1, 0]
	        last_row = 0
	        for c in s:
	            w = widths[ord(c)-97]
	            if last_row + w <= 100:
	                last_row += w
	            else:
	                last_row = w
	                result[0] += 1
	        result[1] = last_row
	        return result
            	      
			
### 运行结果

	Runtime: 20 ms, faster than 67.53% of Python online submissions for Number of Lines To Write String.
	Memory Usage: 13.4 MB, less than 90.91% of Python online submissions for Number of Lines To Write String.


原题链接：https://leetcode.com/problems/number-of-lines-to-write-string/



您的支持是我最大的动力
