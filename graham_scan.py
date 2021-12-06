import matplotlib.pyplot as plt

class Stack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.elems = [None] * capacity
        self.i = -1

    def is_empty(self):
        return len(self.elems) == 0

    def push(self, x):
        self.i += 1
        self.elems[self.i] = x

    def pop(self):
        self.i -= 1
        return self.elems[self.i + 1]

    def top(self):
        return self.elems[self.i]

    def previous_top(self):
        return self.elems[self.i-1]

    def to_list(self):
        return self.elems[0:self.i]

def graham_scan(points, plot=False):
    points.append(points[0])

    s = Stack(capacity = len(points))
    s.push(points[0])
    s.push(points[1])
    s.push(points[2])

    for i in range(3, len(points)):
        top, previous_top = s.top(), s.previous_top()
        while turn(previous_top, top, points[i]) != -1:
            s.pop()
            top, previous_top = s.top(), s.previous_top()
        s.push(points[i])

    print(s.to_list())

    # Plot
    if plot==True:
        xs, ys = zip(*points)
        plt.scatter(xs, ys, color="black")
        xs, ys = zip(*(s.to_list() + [s.to_list()[0]]))
        plt.plot(xs, ys, color="red")
        plt.show()

def turn(p1, p2, p3):
    """
    0 if points are colinear
    1 if points define a left-turn
    -1 if points define a right-turn
    """
    # Compute the z-coordinate of the vectorial product p1p2 x p2p3
    z = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0]- p1[0])
    return 0 if z == 0 else int(z / abs(z))

if __name__ == "__main__":
    points = [(6,1), (2,5), (4,4), (4,7), (6, 5), (7,7), (9, 5), (11,6), (9,3), (8,1)]
    graham_scan(points, True)
