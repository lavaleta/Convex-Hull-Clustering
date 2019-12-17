from builtins import print

import numpy as np
import copy
import math
from tkinter import Tk, Canvas
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import matplotlib.pyplot as plt
from scipy.spatial import distance

from scipy.spatial import ConvexHull, convex_hull_plot_2d


points = np.random.rand(10, 2)                                                                                         # 10 random points in 2-D

points = np.array([[0.36678149, 0.38106586],
        [0.64982347, 0.40332725],
        [0.85628276, 0.93774736],
        [0.88801382, 0.4420135 ],
        [0.26932688, 0.77715105],
        [0.89799713, 0.81355888],
        [0.64361054, 0.43702148],
        [0.52702399, 0.02006482],
        [0.36317455, 0.76319172],
        [0.78053307, 0.33781986]])
# [0.85628276, 0.93774736]  zamenjeno je mosto drugog i treceg clana
# [0.52702399, 0.02006482]
# points = np.random.rand(10, 2)

i = 0
counter = 0
hull_left = None
hull_right = None
hull_left_ver = np.empty([0,2])
hull_right_ver = np.empty([0,2])
min_distance = math.inf
max_distance = -1
what_hull = np.array([2, 1])
normal = np.empty([0,2])
extreme_left_hull = np.empty([0,2])
extreme_right_hull = np.empty([0,2])
x_green = 0
y_green = 0
# plt.plot(points[:,0], points[:,1], 'o')
#
# plt.plot([0.39678149, 0.64982347], [0.40106586, 0.40332725])
# print(points[0:2,0], points[0:2,1])
# plt.show()
#
# breakpoint()

while i < points.size/2:                                                                                               # iterating through all points
    j = i+1
    while j < points.size/2:                                                                                           # for every point we will iterate through the rest of the points
                                                                                                                       # this way we have all the possible line segments
                                                                                                                       # we can say j=i+1 because there is no need to test a line segment in both directions
                                                                                                                       # that way we improve the speed of the algorithm
        k = 0;
        points_left = np.empty([0,2])
        points_right = np.empty([0,2])
        while k < points.size/2:                                                                                       # now for every line segment we iterate through all points
            pom = 0

            if k == i or k == j:                                                                                       # excluding the two points that form the line segment
                k+=1
                continue
            location = ((points[j,0]-points[i,0])*(points[k,1]-points[i,1])                                            # we use the cross product to see where the point 'k' is in relation to the line segment
                        - (points[j,1]-points[i,1])*(points[k,0]-points[i,0]))

            if location > 0:                                                                                           # if location > 0 then the point is to the left of the line segment
                # print(points[k], "ovo je levo od", points[i], points[j])
                points_left = np.append(points_left, [points[k]], axis=0)
            elif location < 0:
                # points_right[pom] =  points[k]
                points_right = np.append(points_right, [points[k]], axis=0)
                pom += 1
            else:                                                                                                      # if location > 0 then the point is on the line segment (it is collinear)
                print("kolinearna je")
                print(points[k], "ovo je kolinearno od", points[i], points[j])

            counter += 1
            k+=1

        left_right = 0

        if(left_right==0):
            points_left_tmp = np.copy(points_left)
            points_left_tmp = np.append(points_left_tmp, [points[i]], axis=0)
            points_left_tmp = np.append(points_left_tmp, [points[j]], axis=0)

            plt.plot(points_left_tmp[:, 0], points_left_tmp[:, 1], 'o')
            if (points_left_tmp.size / 2 > 2):
                hull_left = ConvexHull(points_left_tmp)
                for simplex in hull_left.simplices:
                    plt.plot(points_left_tmp[simplex, 0], points_left_tmp[simplex, 1], 'k-')

            plt.plot(points_right[:, 0], points_right[:, 1], 'o')
            if (points_right.size / 2 > 2):
                hull_right = ConvexHull(points_right)
                for simplex in hull_right.simplices:
                    plt.plot(points_right[simplex, 0], points_right[simplex, 1], 'k-')
            left_right+=1

            # if(points_left.size / 2 > 0 and points_right.size / 2 > 2):
            #     iterator = 0
            #     print("ovo radi")
            #     # while iterator == 0:
            #     for i in range(len(hull_right.vertices)):
            #         print(len(hull_right.vertices))
        # print(hull_left.points)
        # print(hull_left.vertices)
        x_array = np.array(points[i][0])
        x_array = np.append(x_array, points[j][0])
        y_array = np.array(points[i][1])
        y_array = np.append(y_array, points[j][1])
        plt.plot(x_array, y_array, color='red')
        plt.axis([0, 1, 0, 1])

        plt.savefig('Convex hulls/plot' + str(i) + str(j) + '-left.png')
        # plt.show()
        # breakpoint()
        plt.clf()

        if(left_right==1):
            points_right = np.append(points_right, [points[i]], axis=0)
            points_right = np.append(points_right, [points[j]], axis=0)

            plt.plot(points_left[:, 0], points_left[:, 1], 'o')
            if (points_left.size / 2 > 2):
                hull_left = ConvexHull(points_left)
                for simplex in hull_left.simplices:
                    plt.plot(points_left[simplex, 0], points_left[simplex, 1], 'k-')

            plt.plot(points_right[:, 0], points_right[:, 1], 'o')
            if (points_right.size / 2 > 2):
                hull_right = ConvexHull(points_right)
                for simplex in hull_right.simplices:
                    plt.plot(points_right[simplex, 0], points_right[simplex, 1], 'k-')
            left_right += 1
            if (points_left.size / 2 > 2 and points_right.size / 2 > 2):
                iterator = 0
                min_right_LHULL = math.inf
                max_left_LHULL = -1
                min_right_RHULL = math.inf
                max_left_RHULL = -1
                point_of_min = -1
                index_left_of_line = -1
                index_right_of_line = -1
                index_curr_left = -1
                normal = [-1.0*(points[j, 1]-points[i, 1]) + points[i, 0], points[j, 0]-points[i, 0] + points[i, 1]]


                for iter in hull_left.vertices:
                    dist_to_line= np.cross(points[i] - points[j], hull_left.points[iter] - points[j]) / np.linalg.norm(points[i] - points[j])
                    dist_to_line = abs(dist_to_line)

                    dist_to_point = math.sqrt(math.pow(hull_left.points[iter, 0] - points[i,0],2) + math.pow(hull_left.points[iter, 1] - points[i,1],2))
                    len_on_line = math.sqrt(math.pow(dist_to_point,2)-math.pow(dist_to_line,2))
                    direction = 0

                    location = ((normal[0] - points[i, 0]) * (hull_left.points[iter, 1] - points[i, 1])
                                - (normal[1] - points[i, 1]) * (hull_left.points[iter, 0] - points[i, 0]))
                    # if j==5: print(hull_left.points[iter], location, len_on_line)
                    if location < 0:
                        if min_right_LHULL > len_on_line:
                            min_right_LHULL = len_on_line
                            index_right_of_line = iter
                            # if j==5: print("ovo je onaj IF", hull_left.points[iter], min_right_LHULL, len_on_line)
                    if location > 0:
                        # if j == 5: print(hull_left.points[iter], "Kako smo ovde usli")
                        if max_left_LHULL < len_on_line:
                            max_left_LHULL = len_on_line
                            index_left_of_line = iter
                # print(index_left_of_line, "ovo radi? what")
                index_curr_left = index_left_of_line
                if index_left_of_line == -1: index_curr_left = index_right_of_line # ovo mozda mora na kraju

                # if j==5: print(hull_left.points[index_right_of_line])
                # if j==5: breakpoint()

                index_curr_right = -1
                index_left_of_line = -1
                tmp_index = -1
                vert_index_right = 0
                vert_index_left = 0
                for iter in hull_right.vertices:
                    if hull_right.points[iter][0] == points[i][0] and hull_right.points[iter][1] == points[i][1]: tmp_index = iter
                    if hull_right.points[iter][0] != points[i][0] and hull_right.points[iter][1] != points[i][1]:
                        dist_to_line = np.cross(points[i] - points[j],
                                        hull_right.points[iter] - points[j]) / np.linalg.norm(points[i] - points[j])
                        dist_to_line = abs(dist_to_line)
                        dist_to_point = math.sqrt(math.pow(hull_right.points[iter, 0] - points[i, 0], 2) + math.pow(
                                        hull_right.points[iter, 1] - points[i, 1], 2))
                        len_on_line = math.sqrt(math.pow(dist_to_point, 2) - math.pow(dist_to_line, 2))

                        location = ((normal[0] - points[i, 0]) * (hull_right.points[iter, 1] - points[i, 1])
                                    - (normal[1] - points[i, 1]) * (hull_right.points[iter, 0] - points[i, 0]))
                    if location > 0:
                        if max_left_RHULL < len_on_line:
                            max_left_RHULL = len_on_line
                            index_left_of_line = iter
                index_curr_right = index_left_of_line
                if index_left_of_line == -1: index_curr_right = tmp_index

                for iter1 in range(len(hull_left.vertices)):
                    if hull_left.vertices[iter1] == index_curr_left:
                        vert_index_left = iter1
                        break


                for iter2 in range(len(hull_right.vertices)):
                    if hull_right.vertices[iter2] == index_curr_right:
                        vert_index_right = iter2
                        break

                test_iterator = 0
                # # print(hull_left.points[index_curr_left], hull_right.points[index_curr_right])
                # plt.plot(hull_left.points[index_curr_left, 0], hull_left.points[index_curr_left, 1],
                #          '-o', color='green')
                # plt.plot(hull_right.points[index_curr_right, 0], hull_right.points[index_curr_right, 1],
                #          '-o', color='green')
                while 1:
                    # break
                    k = hull_left.vertices[(vert_index_left+1)%len(hull_left.vertices)]


                    location = ((hull_left.points[index_curr_left, 0] - hull_right.points[index_curr_right, 0])*
                                 (hull_left.points[k, 1]  - hull_right.points[index_curr_right, 1])
                                - (hull_left.points[index_curr_left, 1] - hull_right.points[index_curr_right, 1])
                                 * (hull_left.points[k, 0]  - hull_right.points[index_curr_right, 1]))
                    # if j==7:
                    #     print(str(i), str(j))
                    #     print(hull_left.points)
                    #     print("Iteriramo\n", hull_left.points[index_curr_left], hull_right.points[index_curr_right])
                    #     breakpoint()

                    dist_curr = math.sqrt(
                        math.pow(hull_left.points[index_curr_left, 0] - hull_right.points[index_curr_right, 0], 2) +
                        math.pow(hull_left.points[index_curr_left, 1] - hull_right.points[index_curr_right, 1], 2))

                    if location < 0:
                        dist_to_next = math.sqrt(math.pow(hull_left.points[k, 0] - hull_right.points[index_curr_right,0],2) +
                                                 math.pow(hull_left.points[k, 1] - hull_right.points[index_curr_right,1],2))
                        # if j==5: print(dist_to_next, dist_curr)
                        if dist_curr > dist_to_next:
                            vert_index_left+=1
                            index_curr_left = hull_left.vertices[vert_index_left % len(hull_left.vertices)]
                        else:
                            # if j==5: print("zasto sad nisi usao ovde")
                            p = -1
                            if vert_index_right - 1 < 0:
                                p = hull_right.vertices[len(hull_right.vertices)-1]
                            else:
                                p = hull_right.vertices[vert_index_right - 1]

                            dist_to_next_two = math.sqrt(
                                math.pow(hull_left.points[index_curr_left, 0] - hull_right.points[p, 0], 2) +
                                math.pow(hull_left.points[index_curr_left, 1] - hull_right.points[p, 1], 2))
                            if dist_curr > dist_to_next_two:
                                # if j==5: print("ovo bi trebalo da je resilo problem", hull_left.points[index_curr_left], hull_right.points[p])
                                vert_index_right -= 1
                                if vert_index_right < 0:
                                    index_curr_right = hull_right.vertices[len(hull_right.vertices) - 1]
                                else:
                                    index_curr_right = hull_right.vertices[vert_index_right]
                                # if j == 5: print("Sta se desava", hull_left.points[index_curr_left], hull_right.points[index_curr_right])



                            else:
                                # if j == 5: print: print(hull_left.points[index_curr_left], dist_to_next_two)
                                plt.plot(hull_left.points[index_curr_left, 0], hull_left.points[index_curr_left, 1],
                                         '-o', color='green')
                                plt.plot(hull_right.points[index_curr_right, 0]+0.01, hull_right.points[index_curr_right, 1],
                                         '-o', color='green')
                                # print("ovo je najkraci", dist_curr, hull_left.points[index_curr_left], hull_right.points[index_curr_right], str(i), str(j))
                                break
                    else:
                        p = -1
                        if vert_index_right - 1 < 0:
                            p = hull_right.vertices[len(hull_right.vertices) - 1]
                        else:
                            p = hull_right.vertices[vert_index_right - 1]
                            dist_to_next_two = math.sqrt(
                                math.pow(hull_left.points[index_curr_left, 0] - hull_right.points[p, 0], 2) +
                                math.pow(hull_left.points[index_curr_left, 1] - hull_right.points[p, 1], 2))
                        if dist_curr > dist_to_next_two:
                            vert_index_right -= 1
                            if vert_index_right < 0:
                                index_curr_right = hull_right.vertices[len(hull_right.vertices) - 1]
                            else:
                                index_curr_right = hull_right.vertices[vert_index_right]
                        else:
                            plt.plot(hull_left.points[index_curr_left,0], hull_left.points[index_curr_left,1], '-o', color='green')
                            plt.plot(hull_right.points[index_curr_right, 0] + 0.01,
                                     hull_right.points[index_curr_right, 1],
                                     '-o', color='green')
                            # print(dist_curr, hull_left.points[index_curr_left], hull_right.points[index_curr_right],
                            #       "Ovo je najkrace takodje...", str(i), str(j))
                            break
                    test_iterator += 1



        x_array = np.array(points[i][0])
        x_array = np.append(x_array, points[j][0])
        y_array = np.array(points[i][1])
        y_array = np.append(y_array, points[j][1])

        plt.plot(x_array, y_array, color='red')
        plt.axis([0, 1, 0, 1])
        plt.savefig('Convex hulls/plot' + str(i) + str(j) + '-right.png')
        # plt.show()
        # print(str(), str(j))

        plt.clf()
        # if(j==5): breakpoint()
        j=j+1
    i = i + 1

print(max_distance, what_hull)
print(counter)
# hull = ConvexHull(points)

# plt.plot(points[:,0], points[:,1], 'o')
# for simplex in hull.simplices:
#     plt.plot(points[simplex, 0], points[simplex, 1], 'k-')
# plt.show()

