class UnionFind:
    def __init__(self, grid):
        self.parent = []
        self.rank = []
        self.count = 0

        m = len(grid)
        n = len(grid[0]) if m > 0 else 0
        for i in range(m):
            for j in range(n):
                if grid[i][j] == '1':
                    self.parent.append(i * n + j)
                    self.rank.append(1)
                    self.count += 1
                else:
                    self.parent.append(-1)
                self.rank.append(0)
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])
        return self.parent[x]
    
    def union(self, x, y):
        rootX = self.find(x)
        rootY = self.find(y)
        if rootX != rootY:
            if self.rank[rootX] > self.rank[rootY]:
                self.parent[rootY] = rootX
            elif self.rank[rootX] < self.rank[rootY]:
                self.parent[rootX] = rootY
            else:
                self.parent[rootY] = rootX
                self.rank[rootX] += 1
            self.count -= 1

    def getCount(self):
        return self.count

def num_islands(grid):

    if not grid:
        return 0

    uf = UnionFind(grid)
    m = len(grid)
    n = len(grid[0]) if m > 0 else 0

    for i in range(m):
        for j in range(n):
            if grid[i][j] == '1':
                if i > 0 and grid[i - 1][j] == '1':
                    uf.union(i * n + j, (i - 1) * n + j)
                if j > 0 and grid[i][j - 1] == '1':
                    uf.union(i * n + j, i * n + j - 1)
                if i < m - 1 and grid[i + 1][j] == '1':
                    uf.union(i * n + j, (i + 1) * n + j)
                if j < n - 1 and grid[i][j + 1] == '1':
                    uf.union(i * n + j, i * n + j + 1)
                
    return uf.getCount()


def print_grid(grid):
    for i in grid:
        print("\t\t", i)


# Driver code
def main():

    # Example grids
    grid1 = [
        ['1', '1', '1'],
        ['0', '1', '0'],
        ['1', '0', '0'],
        ['1', '0', '1']
    ]

    grid2 = [
        ['1', '1', '1', '1', '0'],
        ['1', '0', '0', '0', '1'],
        ['1', '0', '0', '1', '1'],
        ['0', '1', '0', '1', '0'],
        ['1', '1', '0', '1', '1']
    ]

    grid3 = [
        ['1', '1', '1', '1', '0'],
        ['1', '0', '0', '0', '1'],
        ['1', '1', '1', '1', '1'],
        ['0', '1', '0', '1', '0'],
        ['1', '1', '0', '1', '1']
    ]

    grid4 = [
        ['1', '0', '1', '0', '1'],
        ['0', '1', '0', '1', '0'],
        ['1', '0', '1', '0', '1'],
        ['0', '1', '0', '1', '0'],
        ['1', '0', '1', '0', '1']
    ]

    grid5 = [
        ['1', '0', '1'],
        ['0', '0', '0'],
        ['1', '0', '1']
    ]

    inputs = [grid1, grid2, grid3, grid4, grid5]
    num = 1
    for i in inputs:
        print(num, ".\tGrid:", sep = "")
        print_grid(i)
        print('\n\t Output :', num_islands(i))
        print('-' * 100)
        num += 1


if __name__ == "__main__":
    main()
