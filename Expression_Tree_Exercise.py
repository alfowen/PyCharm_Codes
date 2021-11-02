class Node:
    def __init__(self, value):
        self.left = None
        self.data = value
        self.right = None

def evaluateExpressionTree(root):
    # Empty Tree Check
    if root is None:
        return 0

    # Check leaf node
    if root.left is None and root.right is None:
        return int(root.data)

    # evaluate left tree
    left_sum = evaluateExpressionTree(root.left)

    # evaluate right tree
    right_sum = evaluateExpressionTree(root.right)

    # check which operation to use
    if root.data == '+':
        return left_sum + right_sum
    elif root.data == '-':
        return left_sum - right_sum
    elif root.data == '*':
        return left_sum * right_sum
    elif root.data == '-':
        return left_sum / right_sum

if __name__ == '__main__':

    root = Node('*')
    root.left = Node('+')
    root.right = Node('+')
    root.left.left = Node('3')
    root.left.right = Node('2')
    root.right.left = Node('4')
    root.right.right = Node('5')
    print(evaluateExpressionTree(root))

# output is 45 i.e. (4+5) * (3+2) = 45
