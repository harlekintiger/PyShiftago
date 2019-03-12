import random

import Board as board
import Game as game

score = 0
color = -1


def initialize(player_color):
    global score
    score = 0

    global color
    color = player_color


def turn():
    while True:
        player_input = input("P{0}s turn: ".format(color)).split()

        if len(player_input) == 0:
            rnd_move = random.randint(0, (7 * 4) - 1)

            if not board.test_for_valid_move(rnd_move):
                continue

            print(rnd_move)
            print()
            print()

            board.insert_marble(color, rnd_move)
            break

        if len(player_input) == 1:
            try:
                input_variation = int(player_input[0])
            except ValueError:
                continue

            if board.test_for_valid_move(input_variation):
                print()
                print()
                print()
                board.insert_marble(color, input_variation)
                break

    point_check_return = board.check_for_points(color)
    global score
    score += int(point_check_return.scoredPoints)
    if point_check_return.isDraw:
        return -1
    if score >= game.points_to_win:
        return 1
    return 0


def output():
    return "Color: {0}; Score: {1}".format(color, score)
