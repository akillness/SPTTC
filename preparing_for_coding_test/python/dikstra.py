import heapq

def dijkstra(graph, start):
    # 초기화: 거리 배열, 우선순위 큐
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    queue = []
    heapq.heappush(queue, (0, start))
    
    while queue:
        current_dist, current_node = heapq.heappop(queue)
        # print(current_node)
        # 이미 처리된 노드는 무시
        if current_dist > distances[current_node]:
            continue
        
        # 인접 노드 탐색
        for adjacent, weight in graph[current_node].items():
            distance = current_dist + weight
            
            # 더 짧은 경로 발견 시 업데이트
            if distance < distances[adjacent]:
                distances[adjacent] = distance
                heapq.heappush(queue, (distance, adjacent))
                #print(adjacent)
    
    return distances

# 예제 그래프 (인접 리스트 형태)
graph = {
    'A': {'B': 8, 'C': 1, 'D': 2},
    'B': {},
    'C': {'B': 5, 'D': 2},
    'D': {'E': 3, 'F': 5},
    'E': {'F': 1},
    'F': {'A': 5}
}

# 실행 예시
print(dijkstra(graph, 'A'))