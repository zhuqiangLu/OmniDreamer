import os 



if __name__ == '__main__':
    with open('./test.txt', 'r') as f:
        lines = f.readlines() 
    
    sorted_line = sorted(lines, key=lambda x: x.split('//'))
    print(sorted_line[:30])

    with open('./test_sort.txt', 'w') as fw:
        for l in sorted_line:
            fw.write(l)