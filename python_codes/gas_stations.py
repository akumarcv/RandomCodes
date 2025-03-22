def gas_station_journey(gas, cost):
    if sum(gas) < sum(cost):
        return -1
    else:
        current_gas = 0
        start = 0
        for i in range(len(gas)):
            current_gas = gas[i] - cost[i] + current_gas
            if current_gas < 0:
                current_gas = 0
                start = i + 1
        return start


def main():
    gas = [
        [1, 2, 3, 4, 5],
        [2, 3, 4],
        [1, 1, 1, 1, 1],
        [1, 1, 1, 1, 10],
        [1, 1, 1, 1, 1],
        [1, 2, 3, 4, 5],
    ]
    cost = [
        [3, 4, 5, 1, 2],
        [3, 4, 3],
        [1, 2, 3, 4, 5],
        [2, 2, 1, 3, 1],
        [1, 0, 1, 2, 3],
        [1, 2, 3, 4, 5],
    ]
    for i in range(len(gas)):
        print(i + 1, ".\tGas = ", gas[i], sep="")
        print("\tCost = ", cost[i])
        print(
            "\n \tThe index of the gas station we can start our journey from is ",
            gas_station_journey(gas[i], cost[i]),
            " (If it's -1, then that \n \tmeans no solution exists)",
            sep="",
        )
        print("-" * 100)


if __name__ == "__main__":
    main()
