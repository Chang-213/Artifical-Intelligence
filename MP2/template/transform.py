
# transform.py
# ---------------
# Licensing Information:  You are free to use or extend this projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to the University of Illinois at Urbana-Champaign
#
# Created by Jongdeog Lee (jlee700@illinois.edu) on 09/12/2018

"""
This file contains the transform function that converts the robot arm map
to the maze.
"""
import copy
from arm import Arm
from maze import Maze
from search import *
from geometry import *
from const import *
from util import *

def transformToMaze(arm, goals, obstacles, window, granularity):
    """This function transforms the given 2D map to the maze in MP1.

        Args:
            arm (Arm): arm instance
            goals (list): [(x, y, r)] of goals
            obstacles (list): [(x, y, r)] of obstacles
            window (tuple): (width, height) of the window
            granularity (int): unit of increasing/decreasing degree for angles

        Return:
            Maze: the maze instance generated based on input arguments.

    """
    alpha_start = arm.getArmLimit()[0][0]
    alpha_offset = alpha_start
    alpha_end = arm.getArmLimit()[0][1]
    beta_start = arm.getArmLimit()[1][0]
    beta_offset = beta_start
    beta_end = arm.getArmLimit()[1][1]
    rows = int((alpha_end - alpha_start)/granularity + 1)
    columns = int((beta_end - beta_start)/granularity + 1)
    map = []
    for x in range(rows):
        map2 = []
        for y in range(columns):
            map2.append(SPACE_CHAR)
        map.append(map2)
    alpha = arm.getArmAngle()[0]
    beta = arm.getArmAngle()[1]
    alpha_in = 0
    alpha_first = arm.getArmLimit()[0][0]
    #print(goals)
    while alpha_first <= alpha_end:
        beta_in = 0
        beta_first = beta_start
        while beta_first <= beta_end:
            arm.setArmAngle((alpha_first, beta_first))
            #if(140<arm.getEnd()[0]<160):
                #print(arm.getEnd())

            if(alpha_first%2 == 0 and granularity%2 == 0 and alpha%2 != 0 and isValueInBetween((alpha_first, alpha_first + granularity), alpha) and beta_first == beta):
                map[alpha_in][beta_in] = START_CHAR
            elif(alpha_first == alpha and beta_first == beta):
                map[alpha_in][beta_in] = START_CHAR
            elif(doesArmTipTouchGoals(arm.getEnd(), goals) == True):
                map[alpha_in][beta_in] = OBJECTIVE_CHAR
            elif(doesArmTouchObjects(arm.getArmPosDist(), obstacles, isGoal=False) == True):
                map[alpha_in][beta_in] = WALL_CHAR
            elif(doesArmTouchObjects(arm.getArmPosDist(), goals, isGoal=True) == True):
                map[alpha_in][beta_in] = WALL_CHAR
            elif(isArmWithinWindow(arm.getArmPos(), window) == False):
                map[alpha_in][beta_in] = WALL_CHAR
            else:
                map[alpha_in][beta_in] = SPACE_CHAR
            beta_in = beta_in + 1
            beta_first = beta_first + granularity
        alpha_in = alpha_in + 1
        alpha_first = alpha_first + granularity


    # alpha_in_start = 0
    # alpha_first_start = arm.getArmLimit()[0][0]
    # if(arm.getArmLimit()[0][0]%2 == 0 and alpha%2 == 1):
    #     alpha_first_start = alpha_first_start + 1
    # while alpha_first_start <= alpha_end:
    #     beta_in_start = 0
    #     beta_first_start = beta_start
    #     while beta_first_start <= beta_end:
    #         arm.setArmAngle((alpha_first_start, beta_first_start))
    #         if(alpha_first_start == alpha and beta_first_start == beta):
    #             map[alpha_in_start][beta_in_start] = START_CHAR
    #         beta_in_start = beta_in_start + 1
    #         beta_first_start = beta_first_start + granularity
    #     alpha_in_start = alpha_in_start + 1
    #     alpha_first_start = alpha_first_start + granularity

    return Maze(map, [alpha_offset, beta_offset], granularity)
