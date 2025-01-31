# A Railroad Ink bot

[Railroad Ink](https://horribleguild.com/product-tag/railroad-ink/) is a game where you draw roads and railways on a grid. You get points for connecting exits, creating unbroken roads and railways, using the center of the board, and you lose points for open paths. The game is played over 7 rounds, and at the start of every round you roll dice to see the roads and railways you must draw.

This is a program that plays that game. It uses [Monte Carlo Tree Search](https://www.youtube.com/watch?v=UXW2yZndl7U) to evaluate moves, and a neural network to evaluate the board state. It uses [tree pruning](https://ceur-ws.org/Vol-2862/paper27.pdf) to reduce the size of the tree it needs to search. I did try to use [Rapid Action Value Estimation](https://www.cs.utexas.edu/~pstone/Courses/394Rspring11/resources/mcrave.pdf) to evaluate the board state, but I think my implementation was bad, and it didn't work very well. 

Inspired by [Tiltak](https://github.com/MortenLohne/tiltak), an AI for the board game Tak.

## Commands

### `play`

Let the AI play some Railroad Ink. By default, it will play a single game, spending a second to evaluate each move. The moves will be printed to the console.

`--count` - The number of games to play. Default is 1.

`-d`, `--duration` - The time to spend evaluating each move, in seconds.

`-i`, `--iterations` - The number of iterations to run the MCTS algorithm. If you specify this, the program will search the tree for the specified number of iterations, instead a specified duration.

### `nn`

Train the AI. Sorta. This is all just a pile of things that are vaguely useful for me as I go along.

`-t`, `--train` - A boolean flag that determines if the program should train the neural network it uses to evaluate moves. Defaults to true.

`-g`, `--generate-training-data` - A boolean flag that determines if the program should run some games to create more data to train on. Defaults to false.

`-l`, `--loop-training` - This boolean flag tells the program to never stop training.
