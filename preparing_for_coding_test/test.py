

# 0,1,2
# edge : [0,1], [1,2], [0,1]

def main():
    edges =[[0,1], [1,2], [0,1]]

    n = []

    for e in edges:
        for t in e:
            if t not in n:
                n.append(t)
    # cols(w), rows(h)
    view = [[0 for _ in range(len(n))] for _ in range(len(edges))]
    print(view)
    for y,e in enumerate(edges):
    
        view[y][e[0]] = 1
        view[y][e[1]] = 1
        
        print(view)

    print(view)



if __name__ == "__main__":
    main()