def max_profit(prices):

    buying_day = 0
    
    max_profit = 0
    
    for i in range(1, len(prices)):
        current_profit = prices[i] - prices[buying_day]
        if current_profit > max_profit:
            
            max_profit = current_profit
        if prices[i] < prices[buying_day]:
            buying_day = i
    return max_profit


def main():
    prices = [
        [7, 1, 5, 3, 6, 4],
        [7, 6, 4, 3, 1],
        [1, 2, 3, 4, 5],
        [7, 1, 5, 3, 6, 4, 9],
        [1, 2, 3, 4, 5, 6],
        [1, 2, 3, 4, 5, 6, 7],
        [7, 6, 5, 4, 3, 2, 1]
    ]

    for i, price in enumerate(prices):
        print(f"{i + 1}. Input array: {price}")
        print(f"\tMaximum profit that can be achieved: {max_profit(price)}")
        print("-" * 100)
        

if __name__ == '__main__':
    main()