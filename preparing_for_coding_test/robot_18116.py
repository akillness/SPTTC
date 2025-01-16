

# N ( 1 <= N < 10^6) 의 정보
# 서로 다른 두 부품이 하나의 로봇 부품임을 알려줌 ex) I 1 2 => robot(i) = [1,2]
# Q 는 현재까지 알고 있는 해당 부품의 개수를 물어봄
# Q는 적어도 한번 나온다.


def main():

    import sys
    N = int(sys.stdin.readline())
    cmd = []

    for _ in range(N):
        cmd.append(sys.stdin.readline().split())

    parent = [x for x in range(10**6 + 1)]
    set_size = [1 for x in range(10**6 + 1)]


    def find_root(x):
        if parent[x] == x:
            return x
        else:
            parent[x] = find_root(parent[x])
            return parent[x]


    def union(a, b):
        pa = find_root(a)
        pb = find_root(b)

        if pa < pb:
            parent[pb] = pa
            set_size[pa] += set_size[pb]
            set_size[pb] = 0
        elif pb < pa:
            parent[pa] = pb
            set_size[pb] += set_size[pa]
            set_size[pa] = 0
        # 서로 이미 루트 노드가 같다면
        else:
            pass


    for query in cmd:
        # 합집합
        if query[0] == 'I':
            union(int(query[1]), int(query[2]))
        # 질문
        elif query[0] == 'Q':
            c = int(query[1])
            pc = find_root(c)
            print(set_size[pc])

    # N = int(input())

    # robot = dict()

    # question_q = []
    # for _ in range(N):
    #     order = list(''.join(input().split()))

    #     if order[0] == 'Q' or order[0] == 'q':
    #         for k,v in robot.items():
    #             if order[1] in v:
    #                 question_q.append(len(robot[k]))
    #                 continue
    #             else:
    #                 question_q.append(1)
    #     else:
            
    #         if len(robot) == 0:
    #             robot[order[0]] = []

    #         if order[0] in robot.keys():
    #             if len(robot[order[0]]) == 0:
    #                 robot[order[0]].extend([order[1],order[2]])
    #             else:
    #                 if order[1] not in robot[order[0]]:
    #                     robot[order[0]].extend(order[1])
                    
    #                 if order[2] not in robot[order[0]]:
    #                     robot[order[0]].extend(order[2])

    # while len(question_q) > 0:
    #     print(question_q.pop)

# 4
# I 1 2
# I 3 2
# Q 1
# Q 4
        
if __name__ == "__main__":
    main()