import Board as board
import Game as game

score = 0
color = -1


def initialize(player_color):
    global score
    score = 0

    global color
    color = player_color


def turn(turn_to_take):
    if not board.test_for_valid_move(turn_to_take):
        print("tried invalid move")
        return -2, -5

    #print("Tfp:", turn_to_take)
    #print()

    board.insert_marble(color, turn_to_take)

    point_check_return, max_occurrence_counter = board.check_for_points(color)
    global score
    score += int(point_check_return.scoredPoints)
    if point_check_return.isDraw:
        return -1, 0
    if score >= game.points_to_win:
        return 1, 20

    if point_check_return.scoredPoints == 0:
        return 0, (max_occurrence_counter - 1) * (max_occurrence_counter - 1)
    return 0, point_check_return.scoredPoints * 10


def output():
    return "Color: {0}; Score: {1} - tensor".format(color, score)
