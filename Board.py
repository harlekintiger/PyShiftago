from collections import namedtuple
import math

board_dimension = 7          # Don't change either unless you understand how CheckForPoints() work.
min_scorable_row_length = 5  # // Don't change either unless you understand how CheckForPoints() work.

points_for_length = {
     5: 2,      # length, points
     6: 5,      # length, points
     7: 10}     # length, points

InputVariation = namedtuple('InputVariation', 'xStart yStart, xDir, yDir')
input_variations = [
    InputVariation(0, 0,  0,  1),
    InputVariation(1, 0,  0,  1),
    InputVariation(2, 0,  0,  1),
    InputVariation(3, 0,  0,  1),
    InputVariation(4, 0,  0,  1),
    InputVariation(5, 0,  0,  1),
    InputVariation(6, 0,  0,  1),

    InputVariation(6, 0, -1,  0),
    InputVariation(6, 1, -1,  0),
    InputVariation(6, 2, -1,  0),
    InputVariation(6, 3, -1,  0),
    InputVariation(6, 4, -1,  0),
    InputVariation(6, 5, -1,  0),
    InputVariation(6, 6, -1,  0),

    InputVariation(6, 6,  0, -1),
    InputVariation(5, 6,  0, -1),
    InputVariation(4, 6,  0, -1),
    InputVariation(3, 6,  0, -1),
    InputVariation(2, 6,  0, -1),
    InputVariation(1, 6,  0, -1),
    InputVariation(0, 6,  0, -1),

    InputVariation(0, 6,  1,  0),
    InputVariation(0, 5,  1,  0),
    InputVariation(0, 4,  1,  0),
    InputVariation(0, 3,  1,  0),
    InputVariation(0, 2,  1,  0),
    InputVariation(0, 1,  1,  0),
    InputVariation(0, 0,  1,  0)]

PointCheckReturn = namedtuple('PointCheckReturn', 'isDraw scoredPoints')


board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]
temp_board = [
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0],
    [0, 0, 0, 0, 0, 0, 0]
]


def initialize():
    global board
    board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]

    global temp_board
    temp_board = [
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0]
    ]

    return board


def test_for_valid_move(input_variation):
    xStart = input_variations[input_variation].xStart
    yStart = input_variations[input_variation].yStart
    xDir   = input_variations[input_variation].xDir
    yDir   = input_variations[input_variation].yDir

    for i in range(board_dimension):
        if board[xStart + i * xDir][yStart + i * yDir] == 0:
            return True
    return False


def insert_marble(new_marble, input_variation):
    xStart = input_variations[input_variation].xStart
    yStart = input_variations[input_variation].yStart
    xDir   = input_variations[input_variation].xDir
    yDir   = input_variations[input_variation].yDir

    copy_board_to_temp_board()

    board[xStart][yStart] = new_marble
    if temp_board[xStart][yStart] == 0:
        return

    for i in range(1, board_dimension):
        board[xStart + i * xDir][yStart + i * yDir] = \
            temp_board[xStart + (i - 1) * xDir][yStart + (i - 1) * yDir]
        if temp_board[xStart + i * xDir][yStart + i * yDir] == 0:
            return


def check_for_points(color):
    max_occurrence_counter = 0
    remaining_zeros = 0

    copy_board_to_temp_board()

    for y in range(board_dimension):
        for x in range(board_dimension):
            if temp_board[x][y] == color:
                temp_board[x][y] = 1
            else:
                if temp_board[x][y] == 0:
                    remaining_zeros += 1
                else:
                    temp_board[x][y] = 0

    for curr_variation in input_variations:
        curr_row = [0, 0, 0, 0, 0, 0, 0]
        occurrence_counter = 0

        for i in range(board_dimension):
            curr_row[i] = temp_board[curr_variation.xStart + i * curr_variation.xDir][curr_variation.yStart + i * curr_variation.yDir]
            if curr_row[i] == 1:
                occurrence_counter += 1
                if occurrence_counter > max_occurrence_counter:
                    max_occurrence_counter = occurrence_counter
            else:
                if occurrence_counter < min_scorable_row_length:
                    occurrence_counter = 0
                else:
                    break

        if occurrence_counter < min_scorable_row_length:
            continue

        # <removing middle marbles>

        for i in range(math.trunc(board_dimension / 2) - 1, 0, -1):    # Winning row has to go through the middle spot
            if curr_row[i - 1] != 0:
                curr_row[i] = 8
            else:
                break

        for i in range(math.trunc(board_dimension / 2), board_dimension - 1):
            if curr_row[i + 1] != 0:
                curr_row[i] = 8
            else:
                break

        for i in range(board_dimension):
            if curr_row[i] == 8:
                board[curr_variation.xStart + i * curr_variation.xDir][curr_variation.yStart + i * curr_variation.yDir] = 0

        # </ removing middle marbles>

        return PointCheckReturn(False, points_for_length[occurrence_counter]), max_occurrence_counter

    if remaining_zeros < 4:
        for y in range(board_dimension):
            for x in range(board_dimension):
                if board[x][y] == 0:
                    return PointCheckReturn(False, 0), max_occurrence_counter
        return PointCheckReturn(True, 0), max_occurrence_counter

    return PointCheckReturn(False, 0), max_occurrence_counter


def copy_board_to_temp_board():
    global board
    global temp_board
    for y in range(board_dimension):
        for x in range(board_dimension):
            temp_board[x][y] = board[x][y]


def get_list_of_legal_moves():
    list_of_legal_moves = []
    for i in range(0, board_dimension * 4):
        if test_for_valid_move(i):
            list_of_legal_moves.append(1)
        else:
            list_of_legal_moves.append(0)
    return list_of_legal_moves


def draw_board():
    for y in range(board_dimension):
        for x in range(board_dimension):
            print(board[x][y], end=" ")
        print()

