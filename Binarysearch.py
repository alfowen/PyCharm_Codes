arr = [2, 5, 3, 7, 4, 1, 6, 10, 8, 9]

# [1,2,3,4,5,6,7,8,9,10]
arr.sort()

len_arr = len(arr) - 1

off_set = 0

x = 4

def binarysearch(arr, off_set, len_arr, x):
    if len_arr > 1:
        mid = off_set + (len_arr - off_set) // 2
        if arr[mid] == x:
            return print(mid)
        elif arr[mid] > x:
            return binarysearch(arr, 1, mid - 1, x)
        else:
            return binarysearch(arr, mid + 1, len_arr, x)
    else:
        return print('Number Not found')

binarysearch(arr, off_set, len_arr, x)