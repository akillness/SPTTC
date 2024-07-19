
# Tree 마을, N 개의 번호로 이루어진 마을(Node)이 Tree로 구성 
# 방향은 없고, 마을과 마을사이를 잇는 N-1개의 길(Edge)
# 길이 이어진 마을은 인접한 마을이라고 함
# 3가지 조건을 만족하는 마을 -> 우수마을
# 1) 우수마을끼리는 인접할 수 없음
# 2) 그냥마을은 적어도 하나의 우수마을과 인접해야 함
# 3) Tree 마을
# 우수마을의 주민의 총 합을 최대로 하는 프로그램


# 우수 마을/ 그냥 마을의 탐색을 통해 경우의 수를 통해 우수마을인 경우 최대 주민의 총합을 구하는 문제
# ㄴ 서브트리의 최대합을 구하는 문제 ***s


def main():
    # input 
    N = int(input())
    people = [0] + list(map(int,input().split()))

    # 2D array로 그래프 생성
    graph = [[] for _ in range(N+1)]

    visited = [False] * (N+1)
    dp = [[0,0] for _ in range(N+1)] # 구하려는 인구수의 합, 우수 마을인 경우/아닌 경우 

    # 그래프 정렬
    for n in range(N-1):
        V,E = list(map(int,input().split()))
        # direction
        graph[V].append(E)
        # bi-direction
        graph[E].append(V)
    
    print(graph)

    def dfs(node):
        # dp[n][0] 은, n 이 우수마을이 아닐때 인구수
        # dp[n][1] 은, n 이 우수마을 일 때 인구수
        visited[node]= True

        dp[node][1] += people[node]
        # print(dp[node][1])
        for child in graph[node]:
            if not visited[child]:
                dfs(child)
                dp[node][0] += max(dp[child][0],dp[child][1])
                # print(dp[node][1])
                dp[node][1] += dp[child][0]

    
    dfs(1)
    print(max(dp[1]))



    # def findBestCity(graph):
    #     cities = []
    #     adj_cites = []

    #     for v,e in graph.items():
    #         if len(e) == 1:
    #             cities.append(v)
        
    #     print(cities)

    #     for v,e in graph.items():
    #         if [ adj for adj in e if adj in cities] :
    #             adj_cites.append(v)

    #     print(adj_cites)

    #     remove_cities = cities + adj_cites

    #     print(remove_cities)
    #     for r in remove_cities:
    #         graph.pop(r)
        
    #     for v in graph.keys():
    #         cities.append(v)
            

    #     total_people = 0
    #     for i in cities:
    #         total_people += people[i-1]
    #         print('{0}, {1}'.format(i,total_people))

    #     print(cities)
    #     print(total_people)
    #     return cities, total_people

    # print(graph)
    # cities, total = findBestCity(graph) 

      
    #print(bfs(graph,1))



if __name__=="__main__":
    main()