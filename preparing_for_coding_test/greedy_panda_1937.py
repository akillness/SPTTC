# Top-down
import sys
sys.setrecursionlimit(2000*2000)
rdln = sys.stdin.readline

dx = [0, 0, 1, -1]
dy = [1, -1, 0, 0]
n = int(rdln())
a = [list(map(int, rdln().split())) for _ in range(n)]
d = [[0]*n for _ in range(n)]

def go(i, j):
    
    if d[i][j] != 0: # 이미 방문한 값은 최적값이므로 그대로 반영
        return d[i][j]
    
    d[i][j] = 1 # 시작한 위치 자체도 1개로 포함되므로, 최소값은 1이다.
    for k in range(4):
        y, x = i +dy[k], j+dx[k]
        if 0 <= x < n and 0 <= y < n:
            if a[i][j] < a[y][x]: # 대나무 개수가 많아야 진행
                d[i][j] = max(d[i][j], go(y, x) + 1)
    return d[i][j]

ans = 0
for i in range(n):
    for j in range(n):
        ans = max(ans, go(i,j))

print(d)
print(ans)