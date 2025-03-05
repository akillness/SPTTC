

# N명 입력 / N-1 판매원 존재
# 멘토링인 경우와 아닌경우 (0,1)
# 시너지 = w*사원 + w*사원 

def main():
    N = int(input())
    tree = [[] for _ in range(N+1)]

    P = [0]+[0]+list(map(int,input().split()))
    W = [0] + list(map(int,input().split()))

    dp = [[0,0] for _ in range(N+1)] # 0 은 멘토링, 1 은 멘토링 아닌

    # tree graph 설정
    for i in range(2,N+1):
        tree[P[i]].append(i)

    print(tree)

    def dfs(node):

        # tree search start(node) -> empty
        for n in tree[node]:
            dfs(n)
            dp[node][0] += max(dp[n][1], dp[n][0]) # 맨토링 일 때, 최대 합
        
        dp[node][1] = dp[node][0] # 최대값 비교를 위한 임시 저장
        # if tree[node] == []  때 다음 step
        for n in tree[node]:
            synerge = W[node]*W[n]
            dp[node][1] = max(dp[node][1],dp[node][0]-max(dp[n][0],dp[n][1])+synerge+dp[n][0])
            # 비교를 위해, 이전까지 값이 맨토링일때의 값보다 커진다면 업데이트
            # dp[n][0] < 이전 노드의 값 (기준값)
            # 변화량은 synerge
        
    
    dfs(1)
    print(max(dp[1]))

# 12
# 1 1 1 2 2 6 7 3 4 10 4
# 5 7 3 4 4 2 4 3 3 3 1 5

if __name__ == "__main__":
    main()