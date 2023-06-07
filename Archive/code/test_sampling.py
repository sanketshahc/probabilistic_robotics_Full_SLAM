def makeWheel(population):
    wheel = []
    total = sum(fitness(p) for p in population)
    top = 0
    for p in population:
        f = fitness(p)/total # weight?
        wheel.append((top, top+f, p))
        top += f
    return wheel

def binSearch(wheel, num):
    mid = len(wheel)//2
    low, high, answer = wheel[mid]
    if low<=num<=high:
        return answer
    elif low > num:
        return binSearch(wheel[mid+1:], num)
    else:
        return binSearch(wheel[:mid], num)

def select(wheel, N):
    stepSize = 1.0/N
    answer = []
    r = random.random()
    answer.append(binSearch(wheel, r))
    while len(answer) < N:
        r += stepSize
        if r>1:
            r %= 1
        answer.append(binSearch(wheel, r))
    return answer