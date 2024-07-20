from collections import defaultdict, deque
from heapq import heappush, heappop
import sys
input = sys.stdin.readline
MAX = float('inf')

# input 으로부터 그래프 생성
def make_road_dict(N, M) :
  S, D = map(int, input().split())
  road_dict = defaultdict(dict)
  rev_road_dict = defaultdict(list)
  
  for _ in range(M) :
    U, V, P = map(int, input().split())
    road_dict[U][V] = P
    rev_road_dict[V].append((U, P))

  return S, D, road_dict, rev_road_dict

# 모든 경로 탐색
def dijkstra(S, D, N, road_dict):
  dist_lst = [MAX]*N
  dist_lst[S] = 0
  q = [(0, S)]

  while q :
    dist, node = heappop(q)
    if node == D :
      break
    if dist_lst[node] < dist :
      continue
    
    for next_node, _dist in road_dict[node].items() :
      next_dist = dist + _dist
      if dist_lst[next_node] > next_dist :
        dist_lst[next_node] = next_dist
        heappush(q, (next_dist, next_node))

  return dist_lst

def remove_min_route(S, D, dist_lst, road_dict, rev_road_dict) :
  remove_lst = list()
  q = deque([D])
  
  while q :
    cur = q.popleft()
    if cur == S :
      continue
    for prev, dist in rev_road_dict[cur] :
      if dist_lst[prev] + dist == dist_lst[cur] and (prev, cur) not in remove_lst :
        remove_lst.append((prev, cur))
        q.append(prev)

  for prev, cur in remove_lst :
    del road_dict[prev][cur]
  
def solve() :
  while True:
    N, M = map(int, input().split())
    if N == M == 0 :
      return
    S, D, road_dict, rev_road_dict =  make_road_dict(N, M)
    dist_lst = dijkstra(S, D, N, road_dict)
    if dist_lst[D] == MAX :
      print(-1)
      continue
    remove_min_route(S, D, dist_lst, road_dict, rev_road_dict)
    dist_lst = dijkstra(S, D, N, road_dict)
    print(dist_lst[D] if dist_lst[D] < MAX else -1)
  
solve()