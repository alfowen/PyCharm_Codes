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
