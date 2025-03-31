from collections import deque
from typing import List


def findOrder(numCourses: int, prerequisites: List[List[int]]) -> List[int]:
    """
    Find valid course ordering using topological sort (Kahn's algorithm).

    Args:
        numCourses: Total number of courses labeled from 0 to numCourses-1
        prerequisites: List of [course, prerequisite] pairs where prerequisite
                     must be taken before course

    Returns:
        List[int]: Valid course ordering that satisfies all prerequisites,
                  or empty list if impossible due to cycle

    Time Complexity: O(V + E) where V is number of vertices (courses) and
                    E is number of edges (prerequisites)
    Space Complexity: O(V) for storing the adjacency list and queue

    Example:
        >>> findOrder(4, [[1,0], [2,0], [3,1], [3,2]])
        [0, 1, 2, 3]  # Can take course 0, then 1 and 2, finally 3
    """
    # Build adjacency list and calculate in-degrees
    hash_map = {k: [] for k in range(numCourses)}  # course -> next courses
    indegrees = {k: 0 for k in range(numCourses)}  # course -> num prerequisites

    # Process prerequisites to build graph
    for prereq in prerequisites:
        child, parent = prereq[0], prereq[1]  # child needs parent first
        hash_map[parent].append(child)  # parent -> child edge
        indegrees[child] += 1  # increment child's prerequisites

    # Initialize queue with all courses having no prerequisites
    queue = deque()
    for k, v in indegrees.items():
        if v == 0:  # Course has no prerequisites
            queue.append(k)

    # Process courses in topological order
    result = []
    while queue:
        vertex = queue.popleft()  # Take next available course
        result.append(vertex)  # Add to result order

        # Process all courses that depend on current course
        for neighbors in hash_map[vertex]:
            indegrees[neighbors] -= 1  # Remove current prerequisite
            if indegrees[neighbors] == 0:  # All prerequisites met
                queue.append(neighbors)

    # Check if valid ordering was found
    if len(result) == numCourses:  # All courses included
        return result
    else:
        return []  # Cycle detected


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
