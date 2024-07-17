
# N과 M이 주어질 때, 다음 조건에 맞는 길이가 M인 수열
# ㄴ 1부터 N까지 자연수 중에서 중복 없이 M개를 고른 수열
# ㄴ 수열은 시간순으로 나열

# ex) 4 4  --> 1 2 3 4 / 1 2 4 3 / 1 3 2 4 / 1 3 4 2 

import itertools

def permustation(lst,M):
    combine_lst = []    
    def dfs(combine):
        if len(combine) == M:
            print(' '.join(map(str,combine)))

            return
        
        for i in lst:
            # 해당 값에 대해 Combine 길이만큼 반복
            if i not in combine: # 해당 값에 대해 Combine 길이만큼 반복
                combine.append(i)
                dfs(combine)
                combine.pop()

    dfs(combine_lst)

def main():
    N,M = map(int,input("입력:").split())
    
    num_list = [i for i in range(1,N+1)]
    permustation(num_list,M)

    # for iter in itertools.permutations(num_list,M):
    #     print(iter.join(''))
    


if __name__ == "__main__":
    main()
