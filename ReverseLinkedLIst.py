class Node:
    def __init__(self, data):
        self.data = data
        self.next = None


class LinkedList:

    def __init__(self):
        self.head = None

    def reverselist(self):
        prev = None
        current = self.head
        while current is not None:
            next = current.next
            current.next = prev
            prev = current
            current = next
        self.head = prev


    def load(self, init_data):
        init_data = Node(init_data)
        init_data.next = self.head
        self.head = init_data

    def printlist(self):
        temp = self.head
        while temp:
            print(temp.data)
            temp = temp.next


listrun = LinkedList()
listrun.load(5)
listrun.load(4)
listrun.load(3)
listrun.load(2)
listrun.load(1)

listrun.reverselist()
listrun.printlist()
