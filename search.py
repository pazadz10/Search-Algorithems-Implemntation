import math
from itertools import permutations
import random
import collections
import copy
from collections import defaultdict


def starting_board():     # initialize your starting board here
   h = 6
   arr = [[0 for x in range(h)] for y in range(h)]
   arr[1][4],arr[2][0],arr[3][2],arr[3][4],arr[5][1] = 1, 1, 1, 1, 1
   arr[0][4], arr[1][5], arr[4][0] = 3, 3, 3
   arr[0][3], arr[0][0], arr[1][3] = 2, 2, 2

   return arr

def goal_board():  # initialize goal board
   h = 6
   arr = [[0 for x in range(h)] for y in range(h)]
   arr[1][4], arr[2][0], arr[3][2], arr[3][4], arr[5][1] = 1, 1, 1, 1, 1
   arr[0][4], arr[1][5], arr[2][2] = 3, 3, 3
   arr[0][3], arr[1][1], arr[2][4] = 2, 2, 2

   return arr

def print_board(board):
  for k in range(7):
      if k==0:
          print("",end="   ")
      else:
          print(k,end="  ")
  print()
  for i in range(6):
     for j in range(6):
        if j == 0:
             print(i+1, end=": ")
        if board[i][j]==1:
            print("@",end="  ")
        elif board[i][j]==0 :
            print(" ",end="  ")
        elif board[i][j] == 2:
            print("*", end="  ")
        elif board[i][j] == 3:
            print("&", end="  ")
     print("")

def checkBoard(board, goalboard):
    kList = getKingsCord(board)
    bList = getBishopsCord(board)
    fieldGoals =getFieldsCord(goalboard)
    fieldBoard = getFieldsCord(board)
    flag ='true'
    if len(kList) != len(bList) and len(fieldBoard) != len(fieldGoals):
        flag ='false'
    return flag

def getKingsCord(board):
    kingsList = []
    for i in range(6):
        for j in range(6):
            if board[i][j] == 2:
                temp = [i, j]
                kingsList.append(temp)
    return kingsList

def getBishopsCord(board):
    kingsList = []
    for i in range(6):
        for j in range(6):
            if board[i][j] == 3:
                temp = [i, j]
                kingsList.append(temp)
    return kingsList

def getFieldsCord(board):
    kingsList = []
    for i in range(6):
        for j in range(6):
            if board[i][j] == 1:
                temp = [i, j]
                kingsList.append(temp)
    return kingsList

def validKingsMoves(y, x, board):

    moves = []
    if y-1 >= 0 and board[y-1][x] == 0:
        moves.append(1)
    if x+1 < len(board) and y-1 >= 0 and board[y - 1][x + 1] == 0:
        moves.append(2)
    if x+1 < len(board) and board[y][x+1] == 0:
        moves.append(3)
    if x+1 < len(board) and y+1 < len(board) and board[y+1][x+1] == 0:
        moves.append(4)
    if y+1 < len(board) and board[y+1][x] == 0:
        moves.append(5)
    if x-1 >= 0 and y+1 < len(board) and board[y + 1][x - 1] == 0:
        moves.append(6)
    if x-1 >= 0 and board[y][x-1] == 0:
        moves.append(7)
    if x-1 >= 0 and y-1 >= 0 and board[y - 1][x - 1] == 0:
        moves.append(8)
    return moves

def validBishopsMoves(y, x, board):
    i, j, k, z = 1, 1, 1, 1
    moves = []
    if y + i < 6 and x + i < 6:
        while (x+i < 6 and y+i < 6) and board[y+i][x+i] == 0 : # ++
                moves.append([y+i, x+i])
                i += 1
    if y - j >= 0 and x + j < 6:
        while (y - j >= 0 and x + j < 6) and board[y - j][x + j] == 0 : # -+
            moves.append([y - j, x + j])
            j += 1
    if y - k >= 0 and x - k >= 0:
        while (y - k >= 0 and x - k >= 0) and board[y - k][x - k] == 0 :   # --
            moves.append([y - k, x - k])
            k += 1
    if y + z < 6 and x - z >= 0: #+-
        while (y + z < 6 and x - z >= 0) and board[y + z][x - z] == 0 : #3
            moves.append([y + z, x - z ])
            z += 1
    return moves

def checkBishopMovesToGoal(bishop,goal):
    i, j, f, k = 0, 0, 0, 0
    y = bishop[0]
    x = bishop[1]
    bolian = 'false'
    while i < 6 and y+i < 6 and x+i < 6:
        temp = [y+i, x+i]
        if temp == goal:
            bolian = 'true'
        i += 1
    while j < 6 and y+j < 6 and x-j >= 0:
        temp = [y+j, x-j]
        if temp == goal:
            bolian = 'true'
        j += 1
    while f < 6 and y-f >= 0 and x+f < 6:
        temp = [y-f, x+f]
        if temp == goal:
            bolian = 'true'
        f += 1
    while k < 6 and y-k >= 0 and x-k >= 0:
        temp = [y-k, x-k]
        if temp == goal:
            bolian = 'true'
        k += 1
    return bolian

def calculateBheuristic(bishop, goals):
    distances = []
    for i in range(len(goals)):
        if bishop == goals[i]:
            distances.append(0)
        elif checkBishopMovesToGoal(bishop,goals[i]) == 'true':
            distances.append(1)
        else:
            distances.append(2)
    return distances
def calculateBmoves(bishop, goal):
    distances = 0

    if bishop == goal:
            distances = 0
    elif checkBishopMovesToGoal(bishop,goal) == 'true':
            distances = 1
    else:
            distances = 2
    return distances

def bHeuristic(bishops, goals):
    distances1 = []
    for i in range(len(bishops)):
        distances1.append(calculateBheuristic(bishops[i], goals))
    arr = vector(distances1)
    return arr


def distances(king, kingGoal):
    distances=[]
    for i in range(len(kingGoal)):
       distances.append(math.dist(king, kingGoal[i]))
    return distances
def vector(distances):
    comb = []
    min1 = 1000
    for i in range(len(distances)):
        comb.append(i)
    perm = permutations(comb)
    for i in list(perm):
        temp = 0
        for j in range(len(distances)):
            temp =temp+ distances[j][i[j]]
        if temp < min1:
            min1 = temp
            templist = i
    return [min1,templist]

def findmin(goals): #find which king is the closest to each goal
   min1=[]
   for i in range(len(goals)):
       min1.append([goals[i].index(min(goals[i])), min(goals[i])])
   if min_A[0] == min_B[0] and min_A[1] < min_B[1]:
        goal2[min_B[0]] = 10000
        min_B = [goal2.index(min(goal2)), min(goal2)]
   elif min_A[0] == min_B[0] and min_A[1] > \
           min_B[1]:
        goal1[min_A[0]] = 10000
        min_A = [goal1.index(min(goal1)), min(goal1)]

   if min_A[0] == min_C[0] and min_A[1] < min_C[1]:
        goal3[min_C[0]] = 10000
        min_C = [goal3.index(min(goal3)), min(goal3)]
   elif min_A[0] == min_C[0] and min_A[1] > min_C[1]:
        goal1[min_A[0]] = 10000
        min_A = [goal1.index(min(goal1)), min(goal1)]
   if min_B[0] == min_C[0] and min_B[1] < min_C[1]:
        goal3[min_C[0]] = 10000
        min_C = [goal3.index(min(goal3)), min(goal3)]
   elif min_B[0] == min_C[0] and min_B[1] > min_C[1]:
        goal2[min_B[0]] = 10000
        min_B = [goal2.index(min(goal2)), min(goal2)]
   arr=[min_A, min_B, min_C]
   return arr

def kMove(num,locY, locX):
    newlocation=[locY, locX]
    if num == 1:
        newlocation[0] -= 1

    elif num == 2:
        newlocation[1] += 1
        newlocation[0] -= 1
    elif num == 3:
        newlocation[1] += 1
    elif num == 4:
        newlocation[0] += 1
        newlocation[1] += 1
    elif num == 5:
        newlocation[0] += 1
    elif num == 6:
        newlocation[1] -= 1
        newlocation[0] += 1
    elif num == 7:
        newlocation[1] -= 1
    elif num == 8:
        newlocation[0] -= 1
        newlocation[1] -= 1
    return  newlocation
def aStarBB(bishop, goal, moves, hValue ,board,c,detail_output, heuristicValue):
    Fn = hValue
    min = bishop
    moves = validBishopsMoves(min[0], min[1], board)
    limit = 0
    while min != goal and len(moves) != 0 and limit != 100:
        for i in range(len(moves)):
            newVal = calculateBmoves(moves[i],goal)
            if  newVal <= Fn:
                Fn = newVal
                minmove = moves[i]
        board[min[0]][min[1]] = 0
        min = minmove
        board[minmove[0]][minmove[1]] = 3
        print("________")
        print("Board", c, ":")
        print_board(board)
        if detail_output == 'true' and c == 2:
            print("Heuristic:", heuristicValue)
        c += 1
        limit += 1
        moves = validBishopsMoves(min[0], min[1], board)
    return c


def aStarK(king, goal, moves, distance ,board,c,detail_output, heuristicValue):
    Fn = distance
    min = king
    moves = validKingsMoves(min[0], min[1], board)
    limit=0
    while min != goal and len(moves) != 0 and limit!=100:
     moves = validKingsMoves(min[0], min[1], board)
     for i in range(len(moves)):
        moveLocation = kMove(moves[i], min[0], min[1])
        newDist = math.dist(moveLocation, goal)
        if newDist < distance and newDist < Fn:
            Fn= newDist
            minmove=moveLocation
     board[min[0]][min[1]] = 0
     min=minmove
     board[minmove[0]][minmove[1]] = 2
     print("________")
     print("Board",c,":")
     print_board(board)
     if detail_output == 'true' and c == 2:
         print("Heuristic:",heuristicValue)
     c += 1
     limit +=1
    return c

def check(moves,goal):
    if goal in moves :
         return 'true'

def aStarB(bishop, goal, board, moves, Fn, bol,detail_output, heuristicValue):
      j = 0
      k = 0
      if bishop == goal:
          bol = 'true'
          return [Fn, bol]

      if check(moves, goal) == 'true':
        board[bishop[0]][bishop[1]] = 0
        board[goal[0]][goal[1]] = 3
        print("________")
        print("Board", Fn, ":")
        print_board(board)
        if detail_output == 'true' and Fn == 2:
            print("Heuristic:", heuristicValue)

        Fn +=1
        bol = 'true'
        return [Fn, bol]
      else:
         while j < len(moves):
            neighbor = moves[j]
            newmoves = validBishopsMoves(neighbor[0], neighbor[1], board)
            if check(newmoves, goal) == 'true':
                board[bishop[0]][bishop[1]] = 0
                board[neighbor[0]][neighbor[1]] = 3
                print("________")
                print("Board", Fn, ":")
                print_board(board)
                if detail_output == 'true' and Fn == 2:
                    print("Heuristic:", heuristicValue)

                board[neighbor[0]][neighbor[1]] = 0
                board[goal[0]][goal[1]] = 3
                Fn += 1
                print("________")
                print("Board", Fn, ":")
                print_board(board)
                if detail_output == 'true' and Fn == 2:
                    print("Heuristic:", heuristicValue)

                Fn += 2
                bol = 'true'
                return [Fn, bol]
            else:
               j += 1


      return [Fn, bol]

def bishopMove(board, Goals, bishops, c,goal_board,detail_output):
    arr = bHeuristic(bishops, Goals)
    bishopsGoals=[]
    for k in range(len(Goals)):
        bishopsGoals.append(Goals[arr[1][k]])
    for i in range(len(bishopsGoals)):
     moves = validBishopsMoves(bishops[i][0], bishops[i][1], board)
     Fn = 0
     bol = 'false'
     j = 0
     goal = bishopsGoals[i]
     bol=aStarB(bishops[i], goal, board, moves, c, bol, goal_board, detail_output)[1]
     ff=len(moves)
     if bol != 'true':
      board[bishops[i][0]][bishops[i][1]] = 0
      while bol == 'false' and j < len(moves) and len(moves) != 0:
       n = moves[j]
       nMoves= validBishopsMoves(n[0], n[1], board)
       result = aStarB(n, goal, board, nMoves, c, bol, goal_board, detail_output)
       bol = result[1]
       Fn = result[0]
       j += 1

def bishopMove1(board, Goals, bishops, c,goal_board,detail_output):
    arr = bHeuristic(bishops, Goals)
    bishopsGoals = []
    for k in range(len(Goals)):
        bishopsGoals.append(Goals[arr[1][k]])
    c = 2
    for j in range(len(bishopsGoals)):
        moves = validBishopsMoves(bishops[j][0],bishops[j][1],board)
        goal1 = bishopsGoals[j]
        distance = calculateBmoves(bishops[j], goal1)
        bishop = bishops[j]
        c = aStarBB(bishop, goal1, moves, distance, board, c ,detail_output, arr[0])
    return c


def  kingMove(board, Goals, detail_output, heuristicValue):
    kingsList = getKingsCord(board)
    distances1 = []
    for i in range(len(Goals)):
        distances1.append(distances(Goals[i], kingsList))  # distances array of king 1 from the goals points
    distancesKings = vector(distances1)
    c = 2
    for j in range(len(distances1)):
        moves = validKingsMoves(kingsList[j][0], kingsList[j][1], board)
        goal1 = Goals[distancesKings[1][j]]
        distance = math.dist(kingsList[j], goal1)
        king = kingsList[j]
        c = aStarK(king, goal1, moves, distance,board, c, detail_output, heuristicValue)
    return c

def heuristic(goalsK, goalsB, kingsList, bishopsList):
    distances1 = []
    for i in range(len(goalsK)):
         distances1.append(distances(goalsK[i], kingsList) )   # distances array of king 1 from the goals points
    distancesKings = vector(distances1)
    distancesBishops= bHeuristic(bishopsList, goalsB)
    heuristicValue = int(distancesKings[0]+distancesBishops[0])
    return heuristicValue

def find_path(starting_board,goal_board,search_method,detail_output):
    if search_method == 1:
        A_heuristic_search(starting_board,goal_board,detail_output)
    elif search_method == 2:
        hillClimbing(starting_board,goal_board,detail_output)
    elif search_method == 3:
        SimulatedAnnealing(starting_board,goal_board,detail_output)
    elif search_method ==4:
        kBeam(starting_board,goal_board,detail_output)
    else :
        genetic(starting_board,goal_board,detail_output)

def A_heuristic_search(starting_board, goal_board,detail_output):
    kingGoal=getKingsCord(goal_board)
    bishopGoal=getBishopsCord(goal_board)
    kingsList=getKingsCord(starting_board)
    bishopList=getBishopsCord(starting_board)
    heuristicValue = heuristic(kingGoal, bishopGoal, kingsList, bishopList)
    c = 1
    print("Board", c, "(starting position):")
    print_board(starting_board)
    c = kingMove(starting_board, kingGoal, detail_output,heuristicValue)
    if c == None:
        c=2
    c=bishopMove1(starting_board,bishopGoal,bishopList,c,goal_board,detail_output)
    if starting_board != goal_board:
        print("No path found.")

def hillClimbing(starting_board, goal_board,detail_output):
    print("Board 1 (starting position):")
    print_board(starting_board)
    kingGoals = getKingsCord(goal_board)
    bishopGoals = getBishopsCord(goal_board)
    kingsList = getKingsCord(starting_board)
    bishopList = getBishopsCord(starting_board)
    herVal = heuristic(kingGoals,bishopGoals,kingsList,bishopList)
    board = copy.deepcopy(starting_board)
    count = 0
    flag = 'false'
    heuristicValue = heuristic(kingGoals, bishopGoals, kingsList, bishopList)
    boardNum = 2
    while count <=5 and board != goal_board:
     if count >0:
         print("Board 1 (starting position):")
         starting_board = makeMutation(starting_board,goal_board)
         print_board(starting_board)
         kingGoals = getKingsCord(goal_board)
         bishopGoals = getBishopsCord(goal_board)
         kingsList = getKingsCord(starting_board)
         bishopList = getBishopsCord(starting_board)
         herVal = heuristic(kingGoals, bishopGoals, kingsList, bishopList)
         board = copy.deepcopy(starting_board)
         boardNum = 2
         flag = 'false'

     while flag == 'false':
        kingPossibleMoves=[]
        bishopPossibleMoves = []
        for i in range(len(kingsList)):
            king = kingsList[i]
            bishop = bishopList[i]
            kingMoves = validKingsMoves(king[0], king[1], board)
            bishopMoves = validBishopsMoves(bishop[0], bishop[1], board)
            kingPossibleMoves.append(kingMoves)
            bishopPossibleMoves.append(bishopMoves)
        minKing = 1000
        indexKing = kingsList[0]
        for j in range(len(kingsList)):
            temp = kingsList[j]
            for k in range(len(kingPossibleMoves[j])):
                newBoard = board.copy()
                moveLocation = kMove(kingPossibleMoves[j][k], temp[0], temp[1])
                newBoard[temp[0]][temp[1]] = 0
                newBoard[moveLocation[0]][moveLocation[1]] = 2
                tempKingList = getKingsCord(newBoard)
                newVal = heuristic(kingGoals,bishopGoals,tempKingList,bishopList)
                if newVal < minKing:
                    minKing = newVal
                    kingMove1 = moveLocation
                    indexKing = temp
                    indexK = j

                newBoard[temp[0]][temp[1]] = 2
                newBoard[moveLocation[0]][moveLocation[1]] = 0

        minBishop = 1000
        indexBishop = bishopList[0]
        for b in range(len(bishopPossibleMoves)):
            temp = bishopList[b]
            for y in range(len(bishopPossibleMoves[b])):
                newBoard = copy.deepcopy(board)
                moveLocation = bishopPossibleMoves[b][y]
                newBoard[temp[0]][temp[1]] = 0
                newBoard[moveLocation[0]][moveLocation[1]] = 3
                tempBishopList = getBishopsCord(newBoard)
                newVal = heuristic(kingGoals,bishopGoals,kingsList,tempBishopList)
                if newVal < minBishop:
                    minBishop= newVal
                    indexBishop = temp
                    bishopMove1 = moveLocation
                    indexJ = b
                newBoard[temp[0]][temp[1]] = 3
                newBoard[moveLocation[0]][moveLocation[1]] = 0
        if minKing <= minBishop:
            if minKing > herVal:
                flag = 'true'
            else:
                board[indexKing[0]][indexKing[1]] = 0
                board[kingMove1[0]][kingMove1[1]] = 2
                kingsList[indexK] = kingMove1
                print("Board",boardNum,":" )
                print_board(board)
                if board==2:
                    print("Heuristic:", herVal)
                boardNum += 1
                herVal = minKing
        else:
            if minBishop > herVal:
              flag = 'true'
            else:
                herVal = minBishop
                board[indexBishop[0]][indexBishop[1]] = 0
                board[bishopMove1[0]][bishopMove1[1]] = 3
                bishopList[indexJ] = bishopMove1
                print("Board",boardNum,":" )
                print_board(board)
                if boardNum == 2:
                    print("Heuristic:", herVal)

                boardNum += 1
     count += 1
    if board != goal_board:
         print("No path found.")

def SimulatedAnnealing(starting_board, goal_board,detail_output):
    print("Board 1 (starting position):")
    print_board(starting_board)
    kingGoals = getKingsCord(goal_board)
    bishopGoals = getBishopsCord(goal_board)
    kingsList = getKingsCord(starting_board)
    bishopList = getBishopsCord(starting_board)
    herVal = heuristic(kingGoals,bishopGoals,kingsList,bishopList)
    board = starting_board
    count = 0
    flag = 'false'
    heuristicValue = heuristic(kingGoals, bishopGoals, kingsList, bishopList)
    temp = 1
    coolingRate = 1.01
    boardNum =2
    while temp < 1000 and board != goal_board:
        kingsList = getKingsCord(board)
        bishopList = getBishopsCord(board)
        choose = 'false'
        while choose == 'false' and temp < 1000:
            temp = temp * coolingRate
            check= 'false'
            k=0
            while check == 'false' and k<200:
              kingPossibleMoves = []
              bishopPossibleMoves = []
              for i in range(len(kingsList)):
                    king = kingsList[i]
                    bishop = bishopList[i]
                    kingMoves = validKingsMoves(king[0], king[1], board)
                    bishopMoves = validBishopsMoves(bishop[0], bishop[1], board)
                    kingPossibleMoves.append(kingMoves)
                    bishopPossibleMoves.append(bishopMoves)
              randKing = random.randrange(len(kingPossibleMoves)) #choose Random king
              randBishop = random.randrange(len(bishopPossibleMoves))
              x=len(bishopPossibleMoves[randBishop])
              y=len(kingPossibleMoves[randKing])
              k +=1
              if x > 0 and y > 0:
                  randMoveKing = random.randrange(len(kingPossibleMoves[randKing]))  # choose random king move
                  randMoveBishop = random.randrange(len(bishopPossibleMoves[randBishop]))
                  check = 'true'
            if k < 200:

              tempKing = kingsList[randKing]
              tempBishop = bishopList[randBishop]
              newBoard = board.copy()
              moveLocationKing = kMove(kingPossibleMoves[randKing][randMoveKing], tempKing[0], tempKing[1]) #make the king move
              newBoard[tempKing[0]][tempKing[1]] = 0
              newBoard[moveLocationKing[0]][moveLocationKing[1]] = 2
              tempKingList = getKingsCord(newBoard)
              newKingVal = heuristic(kingGoals, bishopGoals, tempKingList, bishopList) #calaculate new val after king move
              newBoard[tempKing[0]][tempKing[1]] = 2
              newBoard[moveLocationKing[0]][moveLocationKing[1]] = 0
              moveLocationBishop = bishopPossibleMoves[randBishop][randMoveBishop]   #make the bishop move
              newBoard[tempBishop[0]][tempBishop[1]] = 0
              newBoard[moveLocationBishop[0]][moveLocationBishop[1]] = 3
              tempBishopList = getBishopsCord(newBoard)
              newBishopVal = heuristic(kingGoals, bishopGoals, kingsList, tempBishopList)
              newBoard[tempBishop[0]][tempBishop[1]] = 3
              newBoard[moveLocationBishop[0]][moveLocationBishop[1]] = 0
             # calaculate new val after Bishop move
              bol = random.randint(0, 1)
              if bol == 0:
                  cost = herVal - newKingVal
                  if cost > 0:
                      board[tempKing[0]][tempKing[1]] = 0
                      board[moveLocationKing[0]][moveLocationKing[1]] = 2
                      print("action", "(", tempKing, ")->", moveLocationKing,": probability:1.0")
                      print("Board", boardNum, ":")
                      print_board(board)
                      if boardNum == 2:
                          print("Heuristic:", herVal)
                      boardNum += 1
                      herVal = newKingVal
                      choose = 'true'
                  else:
                      delta = math.exp(cost*temp/herVal)
                      if delta > random.random():
                          board[tempKing[0]][tempKing[1]] = 0
                          board[moveLocationKing[0]][moveLocationKing[1]] = 2
                          print("action", "(", tempKing, ")->", moveLocationKing,": probability:", delta)
                          print("Board", boardNum, ":")
                          print_board(board)
                          if boardNum == 2:
                              print("Heuristic:", herVal)
                          boardNum += 1
                          choose = 'true'
                          herVal = newKingVal
                      else:
                          print("action", "(", tempKing, ")->", moveLocationKing,": probability:", delta)
              else:
                  cost = herVal - newBishopVal
                  if cost >= 0:
                      board[tempBishop[0]][tempBishop[1]] = 0
                      board[moveLocationBishop[0]][moveLocationBishop[1]] = 3
                      print("action", "(", tempBishop, ")->", moveLocationBishop,": probability: 1.0", )
                      print("Board", boardNum, ":")
                      print_board(board)
                      if boardNum == 2:
                          print("Heuristic:", herVal)
                      boardNum += 1
                      choose = 'true'
                      herVal = newBishopVal
                  else:
                      delta = math.exp(cost * temp / herVal)
                      if delta > random.random():
                          board[tempBishop[0]][tempBishop[1]] = 0
                          board[moveLocationBishop[0]][moveLocationBishop[1]] = 3
                          print("action", "(", tempBishop, ")->", moveLocationBishop,": probability:", delta)
                          print("Board", boardNum, ":")
                          print_board(board)
                          if boardNum == 2:
                              print("Heuristic:", herVal)

                          boardNum += 1
                          choose = 'true'
                          herVal = newBishopVal
                      else:
                          print("action", "(", tempBishop, ")->",  moveLocationBishop,": probability:", delta)
    if board != goal_board:
         print("No path found.")


def kBeamStart(starting_board, goal_board):
    kingGoals = getKingsCord(goal_board)
    bishopGoals = getBishopsCord(goal_board)
    kingsList = getKingsCord(starting_board)
    bishopList = getBishopsCord(starting_board)
    herVal = heuristic(kingGoals, bishopGoals, kingsList, bishopList)
    board = starting_board.copy()
    board1 = copy.deepcopy(board)
    board2 = copy.deepcopy(board)
    board3 = copy.deepcopy(board)
    count = 0
    kingPossibleMoves = []
    bishopPossibleMoves = []
    for i in range(len(kingsList)):
        king = kingsList[i]
        bishop = bishopList[i]
        kingMoves = validKingsMoves(king[0], king[1], board)
        bishopMoves = validBishopsMoves(bishop[0], bishop[1], board)
        kingPossibleMoves.append(kingMoves)
        bishopPossibleMoves.append(bishopMoves)
    beams = {}
    lenMoves = 0
    for j in range(len(kingsList)):
        arr = []
        temp = kingsList[j]
        for k in range(len(kingPossibleMoves[j])):
            newBoard = board.copy()
            moveLocation = kMove(kingPossibleMoves[j][k], temp[0], temp[1])
            newBoard[temp[0]][temp[1]] = 0
            newBoard[moveLocation[0]][moveLocation[1]] = 2
            tempKingList = getKingsCord(newBoard)
            newVal = heuristic(kingGoals, bishopGoals, tempKingList, bishopList)
            tempVal = [temp, moveLocation]
            arr.append(tempVal)
            beams[newVal] = arr
            newBoard[temp[0]][temp[1]] = 2
            newBoard[moveLocation[0]][moveLocation[1]] = 0

    for b in range(len(bishopPossibleMoves)):
        arr = []
        temp = bishopList[b]
        for y in range(len(bishopPossibleMoves[b])):
            newBoard = board.copy()
            moveLocation = bishopPossibleMoves[b][y]
            newBoard[temp[0]][temp[1]] = 0
            newBoard[moveLocation[0]][moveLocation[1]] = 3
            tempBishopList = getBishopsCord(newBoard)
            newVal = heuristic(kingGoals, bishopGoals, kingsList, tempBishopList)
            tempVal = [temp, moveLocation]
            arr.append(tempVal)
            beams[newVal] = arr
            newBoard[temp[0]][temp[1]] = 3
            newBoard[moveLocation[0]][moveLocation[1]] = 0
    od = collections.OrderedDict(sorted(beams.items()))
    listOd = od.items()
    choosenBeams = []
    beams3= []
    nextBeams = {}
    for d in listOd:
        be = d
        candidete = d[1]
        for o in range(len(candidete)):
            if len(choosenBeams) < 3:
                choosenBeams.append(candidete[o])
    if len(choosenBeams) > 0:
        tool1 = choosenBeams[0][0]
        move = choosenBeams[0][1]
        board1 = makeMove(tool1, move, board1)
        beams3.append(board1)


    if len(choosenBeams) > 1:
        tool2 = choosenBeams[1][0]
        move = choosenBeams[1][1]
        board2 = makeMove(tool2, move, board2)
        beams3.append(board2)

    if len(choosenBeams) > 2:
        tool3 = choosenBeams[2][0]
        move = choosenBeams[2][1]
        board3 = makeMove(tool3, move, board3)
        beams3.append(board3)
    return beams3


def mergeDictionary(dict_1, dict_2):
   dict_3 = {**dict_1, **dict_2}
   for key, value in dict_3.items():
       if key in dict_1 and key in dict_2:
               dict_3[key] = [value , dict_1[key]]
   return dict_3

def chooesBest3(valueList):
    choosenBeams = []
    for d in valueList:
        candidete = d
        for o in range(len(candidete)):
            if len(choosenBeams) < 3:
                choosenBeams.append(candidete[o])
    return choosenBeams
def find3Beams(best1,best2,best3):
    one = copy.deepcopy(best1)
    two = copy.deepcopy(best2)
    three = copy.deepcopy(best3)
    result = []
    if len(one)+len(best2)+len(best3)>0:
     for i in range(3):
        min1 = min(one)
        min2 = min(two)
        min3 = min(three)
        arr=[min1,min2,min3]
        min_index = arr.index(min(arr))
        result.append(min_index)
        if min_index== 0:
            one.remove(min1)
        elif min_index== 1:
            two.remove(min2)
        else:
            three.remove(min3)
    return result

def makeMove(tool,move,board):
    board1 = copy.deepcopy(board)
    board1[move[0]][move[1]] = board1[tool[0]][tool[1]]
    board1[tool[0]][tool[1]] = 0
    return board1
def calMyher(board,goal_board):
    kList= getKingsCord(board)
    bList = getBishopsCord(board)
    kGoals = getKingsCord(goal_board)
    bGoals = getBishopsCord(goal_board)
    val = heuristic(kGoals,bGoals,kList,bList)
    return val

def kBeam(starting_board, goal_board,detail_output):
    print("Board 1 (starting position):")
    print_board(starting_board)
    kingGoals = getKingsCord(goal_board)
    bishopGoals = getBishopsCord(goal_board)
    kingsList = getKingsCord(starting_board)
    bishopList = getBishopsCord(starting_board)
    herVal = heuristic(kingGoals, bishopGoals, kingsList, bishopList)
    board = starting_board.copy()
    board1 = copy.deepcopy(board)
    board2 = copy.deepcopy(board)
    board3 = copy.deepcopy(board)
    numBoard = 2
    count = 0
    kingPossibleMoves = []
    bishopPossibleMoves = []
    for i in range(len(kingsList)):
        king = kingsList[i]
        bishop = bishopList[i]
        kingMoves = validKingsMoves(king[0], king[1], board)
        bishopMoves = validBishopsMoves(bishop[0], bishop[1], board)
        kingPossibleMoves.append(kingMoves)
        bishopPossibleMoves.append(bishopMoves)
    beams = {}
    lenMoves = 0
    for j in range(len(kingsList)):
        arr = []
        temp = kingsList[j]
        for k in range(len(kingPossibleMoves[j])):
            newBoard = board.copy()
            moveLocation = kMove(kingPossibleMoves[j][k], temp[0], temp[1])
            newBoard[temp[0]][temp[1]] = 0
            newBoard[moveLocation[0]][moveLocation[1]] = 2
            tempKingList = getKingsCord(newBoard)
            newVal = heuristic(kingGoals, bishopGoals, tempKingList, bishopList)
            tempVal = [temp,moveLocation]
            arr.append(tempVal)
            beams[newVal] = arr
            newBoard[temp[0]][temp[1]] = 2
            newBoard[moveLocation[0]][moveLocation[1]] = 0

    for b in range(len(bishopPossibleMoves)):
        arr = []
        temp = bishopList[b]
        for y in range(len(bishopPossibleMoves[b])):
            newBoard = board.copy()
            moveLocation = bishopPossibleMoves[b][y]
            newBoard[temp[0]][temp[1]] = 0
            newBoard[moveLocation[0]][moveLocation[1]] = 3
            tempBishopList = getBishopsCord(newBoard)
            newVal = heuristic(kingGoals, bishopGoals, kingsList, tempBishopList)
            tempVal = [temp, moveLocation]
            arr.append(tempVal)
            beams[newVal] = arr
            newBoard[temp[0]][temp[1]] = 3
            newBoard[moveLocation[0]][moveLocation[1]] = 0
    od = collections.OrderedDict(sorted(beams.items()))
    listOd= od.items()
    choosenBeams = []
    nextBeams={}
    oldvalues=[]
    for d in listOd:
        be = d
        candidete = d[1]
        for o in range(len(candidete)):
            if len(choosenBeams) < 3:
                choosenBeams.append(candidete[o])
    if len(choosenBeams) > 0:
        tool1 = choosenBeams[0][0]
        move = choosenBeams[0][1]
        board1 = makeMove(tool1, move, board1)
        nextBeams1 = kBeamStart(board1,goal_board)
        print("-----")
        print("Board",numBoard,"a:")
        print_board(board1)
        val01 = calMyher(board1, goal_board)
        oldvalues.append(val01)

    if len(choosenBeams) > 1:
        tool2 = choosenBeams[1][0]
        move = choosenBeams[1][1]
        board2 = makeMove(tool2,move, board2 )
        print("-----")
        print("Board",numBoard,"b:")
        print_board(board2)
        val02 = calMyher(board2, goal_board)
        oldvalues.append(val02)

    if len(choosenBeams) > 2:
        tool3= choosenBeams[2][0]
        move = choosenBeams[2][1]
        board3 = makeMove(tool3,move,board3)
        print("-----")
        print("Board",numBoard,"c:")
        print_board(board3)
        val03 = calMyher(board3, goal_board)
        oldvalues.append(val03)
    flag = 'true'
    while board1 != goal_board and board2 != goal_board and board3 != goal_board and flag =='true':
        if numBoard == 2500:
            flag = 'false'
        numBoard +=1
        nextBeams11 = kBeamStart(board1, goal_board)
        best1=[]
        for i in nextBeams11:
            val=calMyher(i,goal_board)
            best1.append(val)
        nextBeams22 = kBeamStart(board2, goal_board)
        best2 = []
        for i in nextBeams22:
            val = calMyher(i, goal_board)
            best2.append(val)


        nextBeams33 = kBeamStart(board3, goal_board)
        best3 = []
        for i in nextBeams33:
            val = calMyher(i, goal_board)
            best3.append(val)
        results = find3Beams(best1,best2,best3)
        newValues=[]
        if len(results) > 0:
            beams1 = choseBoard(results[0],best1,best2,best3)
            beams2 = choseBoard(results[0],nextBeams11,nextBeams22,nextBeams33)
            min_index = beams1.index(min(beams1))
            beams1.remove(min(beams1))
            board1 = beams2[min_index]
            print("-----")
            print("Board",numBoard,"a:")
            print_board(board1)
            val11=calMyher(board1,goal_board)
            newValues.append(val11)

        if len(results) > 1:
            beams1 = choseBoard(results[1], best1, best2, best3)
            beams2 = choseBoard(results[1], nextBeams11, nextBeams22, nextBeams33)
            min_index = beams1.index(min(beams1))
            beams1.remove(min(beams1))
            board2 = beams2[min_index]
            print("-----")
            print("Board",numBoard,"b:")
            print_board(board2)
            val22 = calMyher(board2, goal_board)
            newValues.append(val22)

        if len(results) > 2:
            beams1 = choseBoard(results[2], best1, best2, best3)
            beams2 = choseBoard(results[2], nextBeams11, nextBeams22, nextBeams33)
            min_index = beams1.index(min(beams1))
            beams1.remove(min(beams1))
            board3 = beams2[min_index]
            print("-----")
            print("Board",numBoard,"c:")
            print_board(board3)
            val33 = calMyher(board3, goal_board)
            newValues.append(val33)
    if board1 != goal_board and board2 != goal_board and board3 != goal_board:
        print("No path found")


def findBoardVal(list,num,board,goalBoard):
    dictionary = {}
    arr = []
    for i in range(len(list)):
        king1 = list[i][0]
        move = list[i][1]
        board[move[0]][move[1]] = board[king1[0]][king1[1]]
        board[king1[0]][king1[1]] = 0
        kList = getKingsCord(board)
        bList = getBishopsCord(board)
        kGoals= getKingsCord(goalBoard)
        bGoals = getBishopsCord(goalBoard)
        arr.append([num,[king1,move]])
        newVal = heuristic(kGoals, bGoals, kList, bList)
        board[king1[0]][king1[1]] = board[move[0]][move[1]]
        board[move[0]][move[1]] = 0
        dictionary[newVal] = arr
    return dictionary
def choseBoard(num, board1, board2,board3):
    if num == 0:
        return board1
    elif num == 1:
        return  board2
    else:
        return  board3
def chooseRandoSons(sons, prob, utility):
    arr = []
    while len(arr)<2:
        arr = []
        son1 = random.randint(0, len(utility)-1)
        son2 = random.randint(0, len(utility)-1)
        if son1!=son2:
         if utility[son1] > prob:
            arr.append(son1)
         if utility[son2] > prob:
            arr.append(son2)
    return arr
def chooseRandomBoards(prob,utility):
    arr = []
    while len(arr) < 2:
        arr = []
        son1 = random.randint(0, len(utility) - 1)
        son2 = random.randint(0, len(utility) - 1)
        if son1 != son2:
            if utility[son1] > prob:
                arr.append(son1)
            if utility[son2] > prob:
                arr.append(son2)
    return arr
def makeCut(board1, board2):

    kings1 = getKingsCord(board1)
    kings2 = getKingsCord(board2)
    bishops1 = getBishopsCord(board1)
    bishops2 = getBishopsCord(board2)
    newBoard1 = copy.deepcopy(board1)
    newBoard2 = copy.deepcopy(board2)
    p = len(kings1)
    cut = random.randint(1, len(kings1)-1)
    for i in range(cut):
        king1 = kings1[i]
        king2 = kings2[i]
        bishop1 = bishops1[i]
        bishop2 = bishops2[i]
        newBoard1[king1[0]][king1[1]] = board2[king2[0]][king2[1]]
        newBoard1[bishop1[0]][bishop1[1]] = board2[bishop2[0]][bishop2[1]]
    cut = random.randint(1, len(kings1) - 1)
    for i in range(cut):
        king1 = kings1[len(kings1) - 1-i]
        king2 = kings2[i]
        bishop1 = bishops1[len(kings1) - 1-i]
        bishop2 = bishops2[i]
        newBoard2[king2[0]][king2[1]] = board1[king1[0]][king1[1]]
        newBoard2[bishop2[0]][bishop2[1]] = board1[bishop1[0]][bishop1[1]]


    return [newBoard1,newBoard2]
def listMoves(board1,goalsBoard):
    board = copy.deepcopy(board1)
    kingsList = getKingsCord(board)
    bishopList = getBishopsCord(board)
    goalsKing = getKingsCord(goalsBoard)
    goalsBishop= getBishopsCord(goalsBoard)
    herVal = heuristic(goalsKing,goalsBishop,kingsList,bishopList)
    kingPossibleMoves = []
    arr = []
    for i in range(len(kingsList)):
        king = kingsList[i]
        bishop = bishopList[i]
        kingMoves = validKingsMoves(king[0], king[1], board)
        bishopMoves = validBishopsMoves(bishop[0], bishop[1], board)
        for i in bishopMoves:
            temp = [bishop,i]
            tempboard = makeMove(temp[0],temp[1],board)
            newval=calMyher(tempboard,goalsBoard)
            #if newval <= herVal:
            arr.append(temp)
        kingPossibleMoves.append(kingMoves)
    lenMoves = 0
    for j in range(len(kingsList)):
        king11 = kingsList[j]
        for k in range(len(kingPossibleMoves[j])):
            newBoard = copy.deepcopy(board)
            moveLocation = kMove(kingPossibleMoves[j][k], king11[0], king11[1])
            temp = [king11, moveLocation]
            tempboard = makeMove(king11,moveLocation,board)
            newval = calMyher(tempboard, goalsBoard)
            #if newval <= herVal :
            arr.append(temp)


    return arr
def makeMutation(board,goalBoard):
    moves = listMoves(board,goalBoard)
    len1 = len(moves)
    rand = random.randint(0,len1-1)
    move = moves[rand]
    new = makeMove(move[0],move[1],board)
    return new

def genetic(starting_board, goal_board,detail_output):
    global list
    beams = defaultdict(list)
    kingGoals = getKingsCord(goal_board)
    bishopGoals = getBishopsCord(goal_board)
    kingsList = getKingsCord(starting_board)
    bishopList = getBishopsCord(starting_board)
    board = copy.deepcopy(starting_board)
    kingPossibleMoves = []
    bishopPossibleMoves = []
    arr = []
    for i in range(len(kingsList)):
        king = kingsList[i]
        bishop = bishopList[i]
        kingMoves = validKingsMoves(king[0], king[1], board)
        bishopMoves = validBishopsMoves(bishop[0], bishop[1], board)
        kingPossibleMoves.append(kingMoves)
        bishopPossibleMoves.append(bishopMoves)

    for j in range(len(kingsList)):
        temp = kingsList[j]
        for k in range(len(kingPossibleMoves[j])):
            newBoard = copy.deepcopy(board)
            moveLocation = kMove(kingPossibleMoves[j][k], temp[0], temp[1])
            newBoard[temp[0]][temp[1]] = 0
            newBoard[moveLocation[0]][moveLocation[1]] = 2
            tempKingList = getKingsCord(newBoard)
            newVal = heuristic(kingGoals, bishopGoals, tempKingList, bishopList)
            tempVal = [temp,moveLocation]
            arr.append(tempVal)
            beams[newVal].append(arr)
            newBoard[temp[0]][temp[1]] = 2
            newBoard[moveLocation[0]][moveLocation[1]] = 0

    for b in range(len(bishopPossibleMoves)):
        temp = bishopList[b]
        for y in range(len(bishopPossibleMoves[b])):
            newBoard = board.copy()
            moveLocation = bishopPossibleMoves[b][y]
            newBoard[temp[0]][temp[1]] = 0
            newBoard[moveLocation[0]][moveLocation[1]] = 3
            tempBishopList = getBishopsCord(newBoard)
            newVal = heuristic(kingGoals, bishopGoals, kingsList, tempBishopList)
            tempVal = [temp, moveLocation]
            arr.append(tempVal)
            beams[newVal].append(arr)
            newBoard[temp[0]][temp[1]] = 3
            newBoard[moveLocation[0]][moveLocation[1]] = 0
    sorted_dict1 = {key: beams[key] for key in sorted(beams)}
    value_list1 = [sorted_dict1[key] for key in sorted_dict1.keys()]
    sonsBoard = []
    random.shuffle(arr)
    sons = []
    utility = []
    pa=[]
    for s in range(len(arr)):
        if len(sons)<10:
            sons.append(arr[s])
            tampBoard = makeMove(arr[s][0],arr[s][1],board)
            val = 1/ calMyher(tampBoard,goal_board)
            utility.append(val)

    lower = min(utility) * 0.8
    upper = max(utility)
    prob = random.uniform(lower, upper)
    count = 0
    mu = 'No'
    while len(sonsBoard) < 10 and count<500:
      randSons = chooseRandoSons(sons,prob,utility)
      son1 = randSons[0]
      son2 = randSons[1]
      board1 = makeMove(sons[son1][0],sons[son1][1],board)
      board2 = makeMove(sons[son2][0],sons[son2][1],board)
      if count == 0:
          u1 = utility[son1]
          u2 = utility[son2]
          print("Starting board 1 (probability of selection from population:",u1,"):")
          print_board(board1)
          print("Starting board 2 (probability of selection from population:",u2,"):")
          print_board(board2)
      newBoard = makeCut(board1,board2)
      for i in newBoard:
         ran = random.uniform(0, 1)
         if ran > 0.9:
              mu ='Yes'
              i = makeMutation(i, goal_board)

         if checkBoard(i, goal_board) == 'true' and len(sonsBoard) < 10 :
              if  count ==0:
                  print("Result board (mutation happened:",mu,"):")
              sonsBoard.append(i)
              print_board(i)
    tempSonsBoard = []
    count +=1
    while goal_board not in sonsBoard and goal_board not in tempSonsBoard and count<500:
        utility = []
        for j in range(len(sonsBoard)):
            k11=sonsBoard[j]
            val = 1/calMyher(sonsBoard[j],goal_board)
            utility.append(val)
        lower = min(utility)*0.5
        upper = max(utility)
        prob = random.uniform(lower, upper)
        tempSonsBoard=[]
        count +=1
        while len(tempSonsBoard) < 10 and goal_board not in tempSonsBoard and count<500:
            count+=1
            randSons = chooseRandomBoards(prob,utility)
            son1 = randSons[0]
            son2 = randSons[1]
            board1 = sonsBoard[son1]
            board2 = sonsBoard[son2]
            newBoard = makeCut(board1, board2)

            for i in newBoard:
                ran = random.uniform(0, 1)
                if ran > 0.8:
                    i = makeMutation(i,goal_board)
                if checkBoard(i,goal_board) == 'true' and len(tempSonsBoard) < 10 and i not in tempSonsBoard :
                    tempSonsBoard.append(i)


        sonsBoard = tempSonsBoard

    if goal_board not in sonsBoard:
        print("path Not Found")







detail_output= 'true'





