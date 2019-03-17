import Board as board
import KerasPlayer as kerasPlayer
import RandomPlayer as randomPlayer

points_to_win = 10
show_in_console = True

players = [None, None]


def initialize(show_game_in_console = True):
    global players
    players = [
        kerasPlayer,
        randomPlayer
    ]
    for ind, play in enumerate(players, start=1):
        play.initialize(ind)

    global show_in_console
    show_in_console = show_game_in_console

    return board.initialize()


def compute_turn_loop(turn_to_make):
    # keras turn

    if show_in_console:
        set_up()

    turn_result, keras_feedback = kerasPlayer.turn(turn_to_make)

    if not turn_result == 0:
        if turn_result == -2:
            return False, keras_feedback
        if turn_result == 1:
            return game_won(kerasPlayer)
        if turn_result == -1:
            return game_draw()

    # opponent turn

    if show_in_console:
        set_up()

    turn_result = players[1].turn()

    if not turn_result == 0:
        if turn_result == 1:
            return game_won(players[1])
        if turn_result == -1:
            return game_draw()

    return False, keras_feedback


def game_won(player):
    print("Player", player.output(), "won!! Congratulations!")
    if player == players[0]:
        return True, players[0].score * 10
    else:
        return True, - (players[1].score * 10)


def game_draw():
    if players[0].score > players[1].score:
        return game_won(players[0])
    if players[1].score > players[0].score:
        return game_won(players[1])

    print("Game ended in a draw.")
    return True, 25


def set_up():
    for player in players:
        print(player.output())
    board.draw_board()


def draw_game():
    print(players[0].output())
    print(players[1].output())
    board.draw_board()
    print()
