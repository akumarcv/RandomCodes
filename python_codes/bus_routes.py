from collections import deque

def minimum_buses(routes, src, dest):
  
  # Create adjacency list for graph.
  adj_list = {}
  for i, stations in enumerate(routes):
    for station in stations:
      if station not in adj_list:
        adj_list[station] = []
      adj_list[station].append(i)

  # Create a queue with initial source and bus count of 0.
  queue = deque()
  queue.append([src,0])
  # Create a set to contain visited routes of bus.
  visited_buses = set()
  
  # Iterate till queue is empty.
  while queue:
    # Pop station and and number of buses taken and store in variables.
    station, buses_taken = queue.popleft()
    # If we have reached the destination station, return number of buses taken.
    if station == dest:
      return buses_taken
    
    # If station is in graph, then iterate over the stations in graph
    # and if it is not already visited, enqueue all the stations in that bus
    # route along with an incremented bus count and mark the bus visited.
    if station in adj_list:
      for bus in adj_list[station]:
        if bus not in visited_buses:
          for s in routes[bus]:
            queue.append([s, buses_taken+1])
          visited_buses.add(bus)                
    
  # If the route is not found, return -1.
  return -1


# Driver code
def main():
  routes = [[[2, 5, 7], [4, 6, 7]], [[1, 12], [4, 5, 9], [9, 19], [10, 12, 13]], [[1, 12], [10, 5, 9], [4, 19], [10, 12, 13]], [[1, 9, 7, 8], [3, 6, 7], [4, 9], [8, 2, 3, 7], [2, 4, 5]], [[1, 2, 3], [4, 5, 6],[7, 8, 9]]]
  src = [2, 9, 1, 1, 4]
  dest = [6, 12, 9, 5, 6]
  
  for i, bus in enumerate(routes):
    print(i+1, ".\tBus Routes: ", bus, sep ="")
    print("\tSource: ", src[i])
    print("\tDestination: ", dest[i])
    print("\n\tMinimum Buses Required: ", minimum_buses(bus, src[i], dest[i]))
    print("-"*100)

if __name__ == '__main__':
    main()
