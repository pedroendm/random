import matplotlib.pyplot as plt

class Stack:
    def __init__(self, capacity):
        self.capacity = capacity
        self.elems = [None] * capacity
        self.i = -1

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
        return self.elems[0:self.i+1]

def graham_scan(points, plot=False):
    """
     Let points = {p1, p2, p3, . . . , pn} be a set of n points in the plane. Assume that p1 is the point with the
     smallest y-value, points is sorted in strictly decreasing order of polar angle w.r.t. p1 (there are no ties), and we
     want to report the vertices of the convex hull in clockwise order (CW), starting from p1.

     Time complexity: Theta(n).

     :param points: the set of n points in the plane  :: [(Int, Int)]
     :param plot: plot the polygon and the point (requires that matplolib is installed) :: Bool (false, by default)
     :return: a stack containing, bottom to top, the convex hull of the points in CW order
    """
    s = Stack(capacity = len(points))
    s.push(points[0])
    s.push(points[1])
    s.push(points[2])

    for p_i in points[3:]:
        while turn(s.previous_top(), s.top(), p_i) != -1:
            s.pop()
        s.push(p_i)

    # Plot
    if plot==True:
        xs, ys = zip(*points)
        plt.scatter(xs, ys, color="black")
        xs, ys = zip(*(s.to_list() + [points[0]]))
        plt.plot(xs, ys, color="red")
        plt.show()

    return s

def turn(p1, p2, p3):
    """
    0 if the points are colinear
    1 if the points define a left-turn
    -1 if the points define a right-turn
    """
    # Compute the z-coordinate of the vectorial product p1p2 x p2p3
    z = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0]- p1[0])
    return 0 if z == 0 else int(z / abs(z))

if __name__ == "__main__":
    points = [(6,1), (2,5), (4,4), (4,7), (6, 5), (7,7), (9, 5), (11,6), (9,3), (8,1)]
    print(graham_scan(points, True).to_list())
