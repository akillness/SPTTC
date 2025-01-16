


# 2304
# ㄴ 기둥 개수와 각 기둥의 w,h 받아서 전체 면적을 구하는 문제
# ㄴ 지붕에 물이 고이지 않기 때문에 최대 높이 기둥 외에는 작아지는 증분
# ㄴ 기둥의 폭은 1, height만큼 부피를 차지함

def main():
    # n = int(input("기둥 개수 : "))
    # pos_list = []
    # # 값 입력
    # for i in range(0,n):
    #     pos = input("기둥 위치 x,y :")
    #     pos = list(map(int,pos.split()))
    #     pos_list.append(pos)

    n = 7
    lst = [[2,4],
                [11,4],
                [15,8],
                [4,6],
                [5,3],
                [8,10],
                [13,6]]
    # 입력 순서 정렬
    lst = sorted(lst,key = lambda x:x[0])
    
    max_coord, max_height = max(lst,key= lambda x:x[1])
    
    def polygon_area(lst):
        result = 0
        prev_coord, prev_height = lst[0]
        for coord, height in lst[1:]:
            if coord == max_coord:
                result += abs(coord-prev_coord)*prev_height
                return result # end point
            if prev_height < height:
                result += abs(coord-prev_coord)*prev_height
                prev_height = height
                prev_coord = coord


    result_area = polygon_area(lst)
    lst.sort(reverse=True)   
    result_area += polygon_area(lst)

    result_area += max_height
    print(result_area)

if __name__ == "__main__":
    main()