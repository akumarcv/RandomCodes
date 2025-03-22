import heapq
def min_refuel_stops(target, start_fuel, stations): 

    current_fuel_capacity = start_fuel
    
    if current_fuel_capacity >=target:
        return 0
    
    i = 0
    stops = 0
    max_heap = []
    while current_fuel_capacity<target :
        while i<len(stations) and stations[i][0]<=current_fuel_capacity: 
            
            heapq.heappush(max_heap, -stations[i][1])
            i+=1
        if not max_heap:
            return -1
        fuel = heapq.heappop(max_heap)
        current_fuel_capacity = current_fuel_capacity - fuel
        stops+=1
    
    # Replace this placeholder return statement with your code

    return stops

def main():
    input = (
              (3, 3, []),
              (59, 14, [[9, 12], [11, 7], [13, 16], [21, 18], [47, 6]]),
              (15, 3, [[2, 5], [3, 1], [6, 3], [12, 6]]),
              (570, 140, [[140, 200], [160, 130], [310, 200], [330, 250]]),
              (1360, 380, [[310, 160], [380, 620], [700, 89], [850, 190],
               [990, 360]])
    )
    num = 1
    for i in input:
        print(num, ".\tStations : ", i[2], sep="")
        print("\tTarget : ", i[0])
        print("\tStarting fuel : ", i[1])
        print("\n\tMinimum number of refueling stops :",
              min_refuel_stops(i[0], i[1], i[2]))
        num += 1
        print("-" * 100, "\n")


if __name__ == "__main__":
    main()
