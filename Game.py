import Board as board
import HumanPlayer as humanPlayer
import RandomPlayer as randomPlayer

points_to_win = 10
show_in_console = True

players = [None, None]


def initialize(show_game_in_console = True):
    global players
    players = [
        humanPlayer,
        randomPlayer
    ]
    for ind, play in enumerate(players, start=1):
        play.initialize(ind)

    board.initialize()

    global show_in_console
    show_in_console = show_game_in_console


def compute_loop():
    for player in players:

        if show_in_console:
            set_up()

        turn_result = player.turn()

        if turn_result == 0:
            continue
        if turn_result == 1:
            return game_won(player)
        if turn_result == -1:
            return game_draw()

    return True


def game_won(player):
    print("Player", player.output(), "won!! Congratulations!")
    return False


def game_draw():
    if players[0].score > players[1].score:
        return game_won(players[0])
    if players[1].score > players[0].score:
        return game_won(players[1])

    print("Game ended in a draw.")
    return False


def set_up():
    for player in players:
        print(player.output())
    board.draw_board()

