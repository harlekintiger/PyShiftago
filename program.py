import Game as game

roundCount = 0

exit_program = False

while not exit_program:

    game.initialize()

    while game.compute_turn_loop():
        roundCount += 1

    print("GAME ENDED")
    while True:
        inp = input("Want to restart? y / n   ")
        if inp == "n":
            exit_program = True
            break
        if inp == "y":
            break


# Attention!! Run model for AI
# To play player against random: in "Game" change "tenorPlayer" to "humanPlayer"
# To player human against tensor: in "Game" change "randomPlayer" to "humanPlayer"
