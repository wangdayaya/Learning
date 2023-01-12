leetcode 2296. Design a Text Editor （python）



### 描述


Design a text editor with a cursor that can do the following:

* Add text to where the cursor is.
* Delete text from where the cursor is (simulating the backspace key).
* Move the cursor either left or right.

When deleting text, only characters to the left of the cursor will be deleted. The cursor will also remain within the actual text and cannot be moved beyond it. More formally, we have that 0 <= cursor.position <= currentText.length always holds.

Implement the TextEditor class:

* TextEditor() Initializes the object with empty text.
* void addText(string text) Appends text to where the cursor is. The cursor ends to the right of text.
* int deleteText(int k) Deletes k characters to the left of the cursor. Returns the number of characters actually deleted.
* string cursorLeft(int k) Moves the cursor to the left k times. Returns the last min(10, len) characters to the left of the cursor, where len is the number of characters to the left of the cursor.
* string cursorRight(int k) Moves the cursor to the right k times. Returns the last min(10, len) characters to the left of the cursor, where len is the number of characters to the left of the cursor.

Follow-up: Could you find a solution with time complexity of O(k) per call?

Example 1:

	
	Input
	["TextEditor", "addText", "deleteText", "addText", "cursorRight", "cursorLeft", "deleteText", "cursorLeft", "cursorRight"]
	[[], ["leetcode"], [4], ["practice"], [3], [8], [10], [2], [6]]
	Output
	[null, null, 4, null, "etpractice", "leet", 4, "", "practi"]
	
	Explanation
	TextEditor textEditor = new TextEditor(); // The current text is "|". (The '|' character represents the cursor)
	textEditor.addText("leetcode"); // The current text is "leetcode|".
	textEditor.deleteText(4); // return 4
	                          // The current text is "leet|". 
	                          // 4 characters were deleted.
	textEditor.addText("practice"); // The current text is "leetpractice|". 
	textEditor.cursorRight(3); // return "etpractice"
	                           // The current text is "leetpractice|". 
	                           // The cursor cannot be moved beyond the actual text and thus did not move.
	                           // "etpractice" is the last 10 characters to the left of the cursor.
	textEditor.cursorLeft(8); // return "leet"
	                          // The current text is "leet|practice".
	                          // "leet" is the last min(10, 4) = 4 characters to the left of the cursor.
	textEditor.deleteText(10); // return 4
	                           // The current text is "|practice".
	                           // Only 4 characters were deleted.
	textEditor.cursorLeft(2); // return ""
	                          // The current text is "|practice".
	                          // The cursor cannot be moved beyond the actual text and thus did not move. 
	                          // "" is the last min(10, 0) = 0 characters to the left of the cursor.
	textEditor.cursorRight(6); // return "practi"
	                           // The current text is "practi|ce".
	                           // "practi" is the last min(10, 6) = 6 characters to the left of the cursor.
	

	



Note:

	1 <= text.length, k <= 40
	text consists of lowercase English letters.
	At most 2 * 10^4 calls in total will be made to addText, deleteText, cursorLeft and cursorRight.


### 解析


根据题意，设计一个类，里面包含了一个初始化函数和四个可以调用的函数，这个场景就像我们平时打字一样：

* TextEditor() 就是初始化一个空字符串
* void addText(string text) 函数就类似在光标的后面可以加入字符
* int deleteText(int k) 函数就类似我们按回退键，将光标左侧的字符进行删除，但是有可能实际删除的字符数量小于 k ，因为字符串长度有限，所以返回实际删除的数量
* string cursorLeft(int k) 类似我们向左边移动光标，并且返回光标左边最多 10 个字符
* string cursorRight(int k) 类似我们向右边移动光标，并且返回光标左边最多 10 个字符

结合例子和实际情况，应该很好理解题意，而且题目还提出了更高的要求，每个函数的调用的时间复杂度为 O(k) 。

其实这道题不像是 Hard 类型的题目，最多是个 Medium ，我们使用两个栈 a 和 b 就能解决这个题目，可以想象光标就在这两个栈的中间（光标其实是个虚拟的字符，不存在也可以解题）。

* 当调用 addText 的时候，我们不断往栈 a 中添加元素
* 当调用 deleteText 的时候，我们不断从栈 a 中弹出元素，可能一直弹到没有，返回实际弹出的元素个数
* 当调用 cursorLeft 的时候，我们 a 栈顶的元素取出来依次压入 b 中，最后返回 a 最上面的最多 10 个字符
* 当调用  cursorLeft 的时候，我们将 b 栈顶的元素取出来依次压入 a 中，最后返回 a 最上面的最多 10 个字符

每个函数的时间复杂度为 O(k) 。

### 解答
				
	class TextEditor:
	    def __init__(self):
	        self.a, self.b = [], []
	
	    def addText(self, text):
	        self.a.extend(list(text))
	
	    def deleteText(self, k):
	        tmp = k
	        while self.a and tmp:
	            self.a.pop()
	            tmp -= 1
	        return k - tmp
	
	    def cursorLeft(self, k):
	        while self.a and k:
	            k -= 1
	            self.b.append(self.a.pop())
	        return ''.join(self.a[-10:])
	
	    def cursorRight(self, k):
	        while self.b and k:
	            k -= 1
	            self.a.append(self.b.pop())
	        return ''.join(self.a[-10:])


            	      
			
### 运行结果

	75 / 75 test cases passed.
	Status: Accepted
	Runtime: 1032 ms
	Memory Usage: 39.2 MB


### 原题链接

https://leetcode.com/contest/weekly-contest-296/problems/design-a-text-editor/

您的支持是我最大的动力
