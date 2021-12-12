import matplotlib.pyplot as plt

def is_point_in_polygon(p, q, plot=False):
    """
    Given the sequence v_1, ..., v_n of the vertices of a n-vertex convex polygon P, in counterclockwise order (CCW),
    we want to check whether a point q belongs to the interior, the boundary or the exterior of P.

    Time complexity: O(log2 n).

    :param p: the vertices of the polygon P :: [(Int, Int)]
    :param q: the point we want to check :: (Int, Int)
    :param plot: plot the polygon and the point (requires that matplolib is installed) :: Bool (false, by default)
    :return: 1, 0 or -1 if the point q belongs to the interior, the boundary or the exterior of P, respectively
    """

    # Plot
    if plot==True:
        xs, ys = zip(*(p + [p[0]]))
        plt.plot(xs, ys, color="black")
        for i in range(len(p)):
            plt.annotate('p' + str(i), (xs[i], ys[i]), color='red')
        plt.plot(q[0], q[1], marker='o', markersize=3, color="blue")
        plt.annotate('q', (q[0], q[1]), color='blue')
        plt.show()

    # Get the tracks. There are n-2 tracks, where n is the number of vertices in the polygon.
    tracks = construct_tracks(p) # The tracks are already in counterclockwise order (CCW).

    # Find the track of the point q, using binary search.
    # Special case: when point is in the segment [p_1, p_2] or in the segment [p_1, p_n]
    if turn (p[0], p[1], q) == 0 or turn (p[0], p[len(p)-1], q) == 0:
        return 0
    track = find_point_track(tracks, q)

    # Check whether the points lies inside of the track
    return is_point_in_track(track, q) if track else -1

def construct_tracks(p):
    return list([(p[0], x, y) for x, y in zip(p[1:], p[2:])]) # [(p0, p1, p2), (p0, p2, p3), (p0, p3, p4), ...]

def turn(p1, p2, p3):
    """
    0 if points are colinear
    1 if points define a left turn
    -1 if points define a right turn
    """
    # Compute the z-coordinate of the vectorial product p1p2 x p2p3
    z = (p2[0] - p1[0]) * (p3[1] - p1[1]) - (p2[1] - p1[1]) * (p3[0]- p1[0])
    return 0 if z == 0 else int(z / abs(z))

def find_point_track(tracks, q):
    a, b = 0, len(tracks) - 1
    while a <= b:
        m = int(a + (b - a) / 2)
        s, rp, lp = tracks[m] # sp=p_1, rp as "right_point" and lp as "left_point" of the track

        turn_rp = turn(s, rp, q)
        turn_lp = turn(s, lp, q)

        if turn_rp == 1 and turn_lp == -1 :
            return tracks[m]
        elif turn_rp == 0: return tracks[m]
        elif turn_lp == 0: return tracks[m+1]
        elif turn_rp == -1:
            b = m - 1
        else: # turn_lp == 1
            a = m + 1

def is_point_in_track(track, q):
    """1, 0 or -1 if the point q belongs to the interior, the boundary or the exterior of P, respectively"""
    s, rp, lp = track
    return turn(rp, lp, q)

if __name__ == "__main__":
    polygon = [(6,1), (9,3), (11,6), (7,7), (4,7), (2,5)]
    print(is_point_in_polygon(polygon, (7, 7), True))
