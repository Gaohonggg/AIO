class Queue():
    def __init__(self,capacity):
        self.que = []
        self.sizemax = capacity

    def is_empty(self):
        if len(self.que)==0:
            return True
        return False

    def is_full(self):
        if len(self.que)==self.sizemax:
            return True
        return False

    def dequeue(self):
        if self.is_empty():
            return -1
        temp = self.que[0]
        self.que.pop(0)
        return temp

    def  enqueue(self,value):
        if self.is_full():
            return -1
        temp = self.que.append(value)

    def front(self):
        if self.is_empty():
            return -1
        return self.que[0]

queue1 = Queue(5)
queue1.enqueue(1)
queue1.enqueue(2)

print(queue1.is_full())
print(queue1.front())
print(queue1.dequeue())
print(queue1.front())
print(queue1.dequeue())
print(queue1.is_empty())