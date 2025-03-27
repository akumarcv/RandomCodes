from collections import deque
from typing import List


def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    hash_map = {k: [] for k in range(numCourses)}
    indegrees = {k: 0 for k in range(numCourses)}

    for prereq in prerequisites:
        child, parent = prereq[0], prereq[1]
        hash_map[parent].append(child)
        indegrees[child] += 1

    queue = deque()

    for k, v in indegrees.items():
        if v == 0:
            queue.append(k)

    result = []
    while queue:
        vertex = queue.popleft()
        result.append(vertex)
        for neighbors in hash_map[vertex]:
            indegrees[neighbors] -= 1
            if indegrees[neighbors] == 0:
                queue.append(neighbors)

    if len(result) == numCourses:
        return result
    else:
        return []


# Driver code
if __name__ == "__main__":
    test_cases = [
        # numCourses, prerequisites, expected_output
        (2, [[1, 0]], [0, 1]),  # Simple chain
        (4, [[1, 0], [2, 0], [3, 1], [3, 2]], [0, 1, 2, 3]),  # Diamond shape
        (1, [], [0]),  # Single course
        (2, [], [0, 1]),  # Two independent courses
        (3, [[0, 1], [1, 2], [2, 0]], []),  # Cycle, should return empty
        (4, [[1, 0], [2, 1], [3, 1]], [0, 1, 2, 3]),  # Tree structure
    ]

    for i, (num_courses, prereqs, expected) in enumerate(test_cases, 1):
        result = findOrder(num_courses, prereqs)
        print(f"\nTest Case {i}:")
        print(f"Number of Courses: {num_courses}")
        print(f"Prerequisites: {prereqs}")
        print(f"Expected Output: {expected}")
        print(f"Actual Output: {result}")

        # Verify the result is valid
        if result:
            # Create a set of completed courses for O(1) lookup
            completed = set()
            valid = True

            # Check if each course's prerequisites are met
            for course in result:
                for prereq in prereqs:
                    if prereq[0] == course and prereq[1] not in completed:
                        valid = False
                        break
                completed.add(course)

            print(f"Valid Topological Sort: {valid}")
        else:
            print("No valid course order found (cycle detected)")

        assert (not result) == (not expected), f"Test case {i} failed!"
