
# N개의 숫자로 된 마을
# M개의 단방향 도로 연결
# 도로i 번째 길을 지나는데 걸리는 시간 T
# ㄴ 입력 : N, M, 시작마을 / 도로 시작 번호, 종료 번호, 시간
# ㄴ 출력 : 최장시간

# dfs , dp 풀이
import heapq
import sys

input = sys.stdin.readline
INF = int(1e9)

def main():  
    v, e, x = map(int, input().split())
    graph = [[] for _ in range(v + 1)]

    for _ in range(e):
        a, b, cost = map(int, input().split())
        graph[a].append((b, cost)) # 노드 가중치로 설정


    def dijkstra(start):
        q = [] # 최단 거리 테이블을 힙으로 구현
        distance = [INF] * (v + 1)

        heapq.heappush(q, (0, start)) # 힙에 가중치 노드로 구현
        distance[start] = 0

        while q:
            dist, now = heapq.heappop(q)

            if distance[now] < dist: # 최소값을 구했는지 확인
                continue

            for node_index, node_cost in graph[now]:
                cost = dist + node_cost

                if distance[node_index] > cost:
                    distance[node_index] = cost
                    heapq.heappush(q, (cost, node_index))

        return distance


    result = 0
    for i in range(1, v + 1):
        go = dijkstra(i)
        back = dijkstra(x)
        result = max(result, go[x] + back[i])

    print(result)

# 4 8 2
# 1 2 4
# 1 3 2
# 1 4 7
# 2 1 1
# 2 3 5
# 3 1 2
# 3 4 4
# 4 2 3

if __name__=="__main__":
    main()