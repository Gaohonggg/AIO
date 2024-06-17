class Stack():
    def __init__(self,capacity):
        self.stk = []
        self.sizemax = capacity

    def is_empty(self):
        if len(self.stk)==0:
            return True
        return False

    def is_full(self):
        if len(self.stk)==self.sizemax:
            return True
        return False

    def pop(self):
        if self.is_empty():
            return -1
        temp = self.stk[-1]
        self.stk.pop(-1)
        return temp

    def push(self,value):
        if self.is_full():
            return -1
        temp = self.stk.append(value)

    def top(self):
        if self.is_empty():
            return -1
        return self.stk[-1]

stack1 = Stack(5)
stack1.push(1)
stack1.push(2)

print(stack1.is_full())
print(stack1.top())
print(stack1.pop())
print(stack1.top())
print(stack1.pop())
print(stack1.is_empty())