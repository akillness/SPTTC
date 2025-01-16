

import sys, heapq

input = sys.stdin.readline
n = int(input())
q = []
for i in range(n):
	num_list = list(map(int, input().split()))
	if not q: # q에 아무것도 없는 첫번째 입력 시
		for num in num_list:
			heapq.heappush(q, num) # min_heap 구조로 q 채우기
	else:
		for num in num_list: # q에 값이 있을 시 늘 q의 길이를 n으로 유지하기
			if q[0] < num: # q 최솟값보다 현재 숫자가 클 경우 n번째로 큰 수가 바뀌어야 하기 때문에
				# print(num)
				heapq.heappush(q, num) # 현재 숫자를 push 하고
				# print(q)
				heapq.heappop(q) # 기존 최솟값 pop
				
# 5
# 12 7 9 15 5
# 13 8 11 19 6
# 21 10 26 31 16
# 48 14 28 35 25
# 52 20 32 41 49
print(q[0])