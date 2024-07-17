
# W, H 인 2차원 블록 사이에 고이는 물의 양
# 입력값 : 2차원 W,H 값
# 입력값2 : 각 W당 H 값
# ㄴ 블록 사이에 고이는 물의 양은 얼마?

def main():

    W,H = list(map(int,input().split()))
    heights = list(map(int,input().split()))

    pos_zip = [[i,h] for i,h in enumerate(heights)]
    # 최대 기둥 사이의 면적에서 중간 기둥값들을 빼주면 됨 : max_height > heights, sum_middle
    max_height = max(pos_zip, key=lambda x:x[1])[1]
    def pool_polygon(lst):
        middle_area = 0
        result = 0
        prev_coord, prev_height = lst[0][0],lst[0][1]
        for coord, height in lst:
            if height == max_height:
                result += prev_height * abs(coord - prev_coord)
                return abs(result - middle_area)
            
            if prev_height < height:
                result += prev_height * abs(coord - prev_coord)
                prev_coord, prev_height = coord, height
            
            # 중간 기둥값 찾기
            middle_area += height
            # print(middle_area)
                

    result = pool_polygon(pos_zip)
    # print(result)
    # result = 0
    pos_zip.sort(reverse=True)
    # print(pos_zip)
    result += pool_polygon(pos_zip)
    

    print(result)

    


if __name__ == "__main__":
    main()