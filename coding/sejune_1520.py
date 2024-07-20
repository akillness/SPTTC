
# N*M 지도
# 왼쪽위에서부터 시작해서 오른쪽 끝으로 도착
# 현재 노드보다 작은 숫자로 이동

def main():
    N,M = list(map(int,input().split()))

    rows = [list(map(int,input().split())) for _ in range(N)] 


    dx = [0,0,-1,1]
    dy = [1,-1,0,0]
     
    dp = [[-1]*M for _ in range(N)]


    def check_condition(x, y, cur):
        return 0 <= x < M and 0 <= y < N and rows[y][x] < cur

    def go(x, y):
        # 맨 마지막 도착시 종료 시점
        if x == M-1 and y == N-1:
            return 1
        
        if dp[y][x] == -1:
            dp[y][x] = 0
            for i in range(4):
                dr_x, dr_y = x + dx[i], y + dy[i]
                if check_condition(dr_x, dr_y, rows[y][x]):
                    dp[y][x] += go(dr_x, dr_y)
        
        return dp[y][x]
            
        
    print(go(0, 0))
               
# 4 5
# 50 45 37 32 30
# 35 50 40 20 25
# 30 30 25 17 28
# 27 24 22 15 10    

if __name__ == "__main__":
    main()