from heapq import *

def k_smallest_pairs(list1, list2, k):
    # storing the length of lists to use it in a loop later
    list_length = len(list1)
    # declaring a min-heap to keep track of the smallest sums
    min_heap = []
    # to store the pairs with smallest sums
    pairs = []

    # iterate over the length of list 1
    for i in range(min(k, list_length)):
        # computing sum of pairs all elements of list1 with first index
        # of list2 and placing it in the min-heap
        heappush(min_heap, (list1[i] + list2[0], i, 0))

    counter = 1

    # iterate over elements of min-heap and only go upto k
    while min_heap and counter <= k:
        # placing sum of the top element of min-heap
        # and its corresponding pairs in i and j
        sum_of_pairs, i, j = heappop(min_heap)

        # add pairs with the smallest sum in the new list
        pairs.append([list1[i], list2[j]])

        # increment the index for 2nd list, as we've
        # compared all possible pairs with the 1st index of list2
        next_element = j + 1

        # if next element is available for list2 then add it to the heap
        if len(list2) > next_element:
            heappush(min_heap,
                     (list1[i] + list2[next_element], i, next_element))

        counter += 1

    # return the pairs with the smallest sums
    return pairs


# Driver code
def main():

    # multiple inputs for efficient results
    list1 = [[2, 8, 9],
             [1, 2, 300],
             [1, 1, 2],
             [4, 6],
             [4, 7, 9],
             [1, 1, 2]]

    # multiple inputs for efficient results
    list2 = [[1, 3, 6],
             [1, 11, 20, 35, 300],
             [1, 2, 3],
             [2, 3],
             [4, 7, 9],
             [1]]

    k = [9, 30, 1, 2, 5, 4]

    # loop to execute till the length of list k
    for i in range(len(k)):
        print(i + 1, ".\t Input pairs: ", list1[i], ", ", list2[i],
              f"\n\t k = {k[i]}", sep="")
        print("\t Pairs with the smallest sum are: ",
              k_smallest_pairs(list1[i], list2[i], k[i]), sep="")
        print("-" * 100)


if __name__ == '__main__':
    main()