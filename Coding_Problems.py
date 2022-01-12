# Given a list of numbers and a number k, return whether any two numbers from the list add up to k.
# For example, given [10, 15, 3, 7] and k of 17, return true since 10 + 7 is 17.

a = [10, 15, 3, 7]
k = 89


def pair_finder(a):
    for i in range(len(a)):
        for j in range(i + 1, len(a)):
            if a[i] + a[j] == k:
                print(f'Return Value is {a[i]} & {a[j]}')
                return True

    if i == len(a) - 1 & j == len(a) - 1:
        print(f'Value pair for {k} is not found in list {a}')


pair_finder(a)


# Given a list of integers, return the largest product that can be made by multiplying any three integers.

# For example, if the list is [-10, -10, 5, 2], we should return 500, since that's -10 * -10 * 5.

x = [-10, -10, 5, 2]


def mul_func(x):
    ind = 0
    for i in range(len(x)):
        for j in range(i+1, len(x)):
            for k in range(j+1, len(x)):
                y = x[i] * x[j] * x[k]
                if y > ind:
                    ind = y
                    z=[]
                    z.append(x[i])
                    z.append(x[j])
                    z.append(x[k])

    print(f'{z} is the max mul pair')
mul_func(x)


# Given a N by M matrix of numbers, print out the matrix in a clockwise spiral
import numpy as np

x = [[1, 2, 3, 4, 5],
     [6, 7, 8, 9, 10],
     [11, 12, 13, 14, 15],
     [16, 17, 18, 19, 20]]

a = np.shape(x)
# print(x[0][0])
# print(x[0][1])
# print(x[0][2])
# print(x[0][3])
# print(x[0][4])
# print(x[1][4])
# print(x[2][4])
# print(x[3][4])
# print(x[3][3])
# print(x[3][2])
# print(x[3][1])
# print(x[3][0])
# print(x[2][0])
# print(x[1][0])
# print(x[1][1])
# print(x[1][2])
# print(x[1][3])
# print(x[2][3])
# print(x[2][2])
# print(x[2][1])




top = 0
bot = a[0] - 1
left = 0
right = a[1] - 1
dir = 0
# left to right is dir 0
# top to bottom is dir 1
# right to left is dir 2
# bot to top is dir 3
counter = 0

while counter < (a[0] * a[1]):
    if dir == 0:
        for i in range(left, right + 1):
            print(x[top][i])
            counter = counter + 1
        top = top + 1

    elif dir == 1:
        for i in range(top, bot + 1):
            print(x[i][right])
            counter = counter + 1
        right = right - 1

    elif dir == 2:
        for i in range(right, left - 1, -1):
            print(x[bot][i])
            counter = counter + 1
        bot = bot - 1

    elif dir == 3:
        for i in range(bot, top - 1, -1):
            print(x[i][left])
            counter = counter + 1
        left = left + 1
    dir = (dir + 1) % 4

# Given a 2D matrix of characters and a target word, write a function that returns whether the word can be found
# in the matrix by going left-to-right, or up-to-down.

x = [['F', 'A', 'C', 'I'],
     ['O', 'B', 'Q', 'P'],
     ['A', 'N', 'O', 'B'],
     ['M', 'A', 'S', 'S']]

k = 'ANOB'

a = np.shape(x)
top = 0
bottom = a[0] - 1
left = 0
right = a[1] - 1
ind = 0
l = []

while ind == 0:
    for j in range(top, bottom + 1):
        for i in range(left, right + 1):
            l.append(x[j][i])
            m = ''.join(l)

        if m == k:
            ind = 2
            print(f'String {k} Found On Left to Right Starting Position {j} ')

        l = []
    if ind == 2:
        pass
    else:
        for p in range(left, right + 1):
            for o in range(top, bottom + 1):
                l.append(x[o][p])
                m = ''.join(l)

            if m == k:
                print(f'String {k} Found On Top To Bottom Starting  Position {p} ')
                ind = 1
            elif (p == a[1] - 1 & o == a[0] - 1):
                ind = 1
            l = []


# Given a mapping of digits to letters (as in a phone number), and a digit string, return all possible letters the
# number could represent. You can assume each valid number in the mapping is a single digit.
#
# For example if {“2”: [“a”, “b”, “c”], 3: [“d”, “e”, “f”], …} then “23” should
# return [“ad”, “ae”, “af”, “bd”, “be”, “bf”, “cd”, “ce”, “cf"].

from collections import deque


def func_word_comb(number, n, table):
    list = []
    # q = deque()
    q = []
    q.append('')
    while len(q) != 0:
        # print(q)
        s = q.pop()
        print(len(s))
        if len(s) == n:

            list.append(s)
        else:
            for letter in table[number[len(s)]]:
                # print(f'first record is {letter}')
                q.append(s + letter)
    r = ''
    for i in list:
        r += i + ' '
    print(r)
    return


number = [2, 3]
n = len(number)
table = ['0', '1', 'abc', 'def', 'ghi', 'jkl', 'mno', 'pqrs', 'tuv', 'wxyz']
func_word_comb(number, n, table)

# What does the below code snippet print out? How can we fix the anonymous functions to behave as we'd expect?

functions = []
for i in range(10):
    functions.append(lambda i=i: i)

for f in functions:
    print(f())


# Given a number in the form of a list of digits, return all possible permutations.
#
# For example, given [1,2,3], return [[1,2,3],[1,3,2],[2,1,3],[2,3,1],[3,1,2],[3,2,1]].


x = [1, 2, 3]
s = 0
l = 3
def rev_function(x,s, l):
     if s == l:
          pass
          print(x)
     else:
          for i in range(s,l+1):
               x[s], x[i] = x[i], x[s]
               rev_function(x, s+1, l)
               x[s], x[i] = x[i], x[s]

rev_function(x,s, l-1)

# Given two strings A and B, return whether or not A can be shifted some number of times to get B.
#
# For example, if A is abcde and B is cdeab, return true. If A is abc and B is acb, return false.
i

# a = 'abcde'
# b = 'cdeab'

a = 'abc'
b = 'acb'

l = len(a)
counter = 0
if len(a) != len(b):
    print('Size of a and b are not the same - this operation cannot be completed')
else:
    for i in range(l-1):
        counter = counter + 1
        b = b[(l - 1):] + b[:(l - 1)]
        print(b)
        print(l)
        print(counter)
        if a == b:
            print(f'String {b} can be shifted to match {a} anticlockwise @ {counter} iteration')
            break
        elif a != b and (l-1) == counter:
            print(f'String {b} cannot be shifted to match {a} anticlockwise @ {counter} iteration')
