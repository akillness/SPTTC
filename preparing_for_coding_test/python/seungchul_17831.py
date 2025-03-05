import sys
sys.setrecursionlimit(10**6)
N = int(sys.stdin.readline())
parent = [0] + [0] + list(map(int, sys.stdin.readline().split())) # 사원수 2<= N <= 200000
W = [0] + list(map(int, sys.stdin.readline().split())) # 0번(첫번째) 제외
tree = [[] for _ in range(N+1)]
dp = [[0, 0] for _ in range(N+1)]
visit = [False] * (N+1)

for i in range(2, N+1):
    tree[parent[i]].append(i)


def dfs(node):
    # visit[node] = True

    print(tree[node])
    for x in tree[node]:
        print(x)
        dfs(x)
        dp[node][0] += max(dp[x][0], dp[x][1])
    
    dp[node][1] = dp[node][0] 
    # node를 멘토링에 넣었을 때, node를 루트로 하는 서브트리의 최대 시너지
    # node를 멘토링에 넣지 않았을 때, node를 루트로 하는 서브트리의 최대 시너지
    for x in tree[node]:
        dp[node][1] = max(dp[node][1], dp[node][0] - max(dp[x][0], dp[x][1]) + (W[node] * W[x]) + dp[x][0])
        

    print(dp)

dfs(1)
print(max(dp[1]))