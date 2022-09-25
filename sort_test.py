import os 



if __name__ == '__main__':
    with open('./train.txt', 'r') as f:
        lines = f.readlines() 
    
    sorted_line = sorted(lines, key=lambda x: x.split('//'))
    print(sorted_line[:30])

    with open('./train_sort.txt', 'w') as fw:
        for l in sorted_line:
            fw.write(l)