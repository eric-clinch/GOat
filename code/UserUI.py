#!/usr/bin/python3

import subprocess
import time
import sys
from pyMCTS.Board import Board
from pyMCTS.MCTS import MCTS
from pyMCTS.NaiveEvaluator import NaiveEvaluator
from pyMCTS.NNEvaluator import NNEvaluatorFactory

SECONDS_PER_MOVE = 10
THREADS = -1 # Use all available threads
BOARD_SIZE = 9

#taken from http://www.cs.cmu.edu/~112/notes/notes-graphics.html
def rgbString(red, green, blue):
    return "#%02x%02x%02x" % (red, green, blue)

# animation framework taken from the 15-112 course website
# https://www.cs.cmu.edu/~112/notes/notes-animations-part2.html

from tkinter import *

def init(data):
    data.board = Board(BOARD_SIZE)
    data.boardLen = BOARD_SIZE
    data.gameOver = False
    data.userWon = False
    data.userPlayers = []
    data.confidence = None

    data.backgroundColor = rgbString(219, 190, 122)
    data.margin = 50
    data.verticalMargin = 20
    data.horizontalMargin = 60
    data.cellWidth = (data.width - 2 * data.verticalMargin) / data.boardLen
    data.cellHeight = (data.height - 2 * data.horizontalMargin) / data.boardLen
    data.circleMargin = 2

    # Where the most recent stone was placed
    data.placedRow = None
    data.placedCol = None

    data.evaluator = NNEvaluatorFactory("trained_models/naive_mcts_model", BOARD_SIZE)

    data.passButton = Button(data.root, text="Pass", relief=GROOVE, command=lambda: makePassMove(data))
    data.passButtonWidth = 100

    data.modelEvaluateButton = Button(data.root, text="AI Evaluate", relief=GROOVE,
                                      command=lambda: EvaluateBoard(data))
    data.evaluatorButtonWidth = 180
    data.evaluator_value = None
    data.evaluator_policy = None

    data.require_redraw = False


def mousePressed(event, data):
    if data.gameOver: return
    if data.board.current_player not in data.userPlayers:
        return
    row = int((event.y - data.horizontalMargin) // data.cellHeight)
    col = int((event.x - data.verticalMargin) // data.cellWidth)
    if (0 <= row < data.boardLen and 0 <= col < data.boardLen):
        if data.board.LegalMove(row, col):
            data.placedRow = row
            data.placedCol = col
            data.board.MakeMove(row, col)
            data.require_redraw = True

            data.evaluator_value, data.evaluator_policy = None, None
            data.gameOver = data.board.IsGameOver()

def keyPressed(event, data):
    # use event.char and event.keysym
    pass

def makePassMove(data):
    if not data.gameOver:
        data.board.MakeMove(-1, -1)

def EvaluateBoard(data):
    data.evaluator_value, data.evaluator_policy, _ = data.evaluator(data.board)

def timerFired(data):
    if data.gameOver or data.require_redraw:
        return
    if data.board.current_player not in data.userPlayers:
        # mcts_move = data.board.GetMCTSMove(THREADS, SECONDS_PER_MOVE)
        mcts_move = MCTS(data.board, data.evaluator, SECONDS_PER_MOVE)
        data.board.MakeMove(mcts_move.row, mcts_move.col)
        data.placedRow = mcts_move.row
        data.placedCol = mcts_move.col
        data.confidence = mcts_move.confidence

        data.evaluator_value, data.evaluator_policy = None, None
        data.gameOver = data.board.IsGameOver()
        data.require_redraw = True


def redrawAll(canvas, data):
    canvas.create_rectangle(0, 0, data.width, data.height,
                            fill=data.backgroundColor)

    top = data.horizontalMargin
    bot = top + data.cellHeight
    board = data.board.BoardList()
    for row in range(data.boardLen):
        for col in range(data.boardLen):
            left = data.verticalMargin + data.cellWidth * col
            right = left + data.cellWidth
            if (row < data.boardLen - 1 and col < data.boardLen - 1):
                canvas.create_rectangle(left + data.cellWidth / 2, 
                    top + data.cellHeight / 2, right + data.cellWidth / 2, 
                    bot + data.cellHeight / 2)
            if board[row][col] != '-':
                color = 'red'
                if board[row][col] == 'B':
                    color = 'black'
                if board[row][col] == 'W':
                    color = 'white'
                outline = 'black'
                width = 1
                if row == data.placedRow and col == data.placedCol:
                    outline = 'red'
                    width = 4
                canvas.create_oval(left + data.circleMargin, 
                    top + data.circleMargin, right - data.circleMargin, 
                    bot - data.circleMargin, fill=color, outline=outline, width=width)

            if data.evaluator_policy is not None:
                prior = data.evaluator_policy[row][col]
                color = 'red'
                radius = (data.cellWidth - 2 * data.circleMargin) * prior / 2
                x_center = (left + right) / 2
                y_center = (top + bot) / 2
                canvas.create_oval(x_center - radius, 
                    y_center - radius, x_center + radius, 
                    y_center + radius, fill=color, width=0)

        top = bot
        bot += data.cellHeight

    if (data.confidence is not None):
        canvas.create_text(data.verticalMargin, data.horizontalMargin / 2, 
                       text = "AI confidence: %.2f" % (data.confidence), anchor=W)

    canvas.create_text(data.verticalMargin, data.height - (data.horizontalMargin) / 3, 
                       text = "White score: %d" % data.board.GetPlayerScore(1), anchor=SW)
    canvas.create_text(data.width / 2, data.height - (data.horizontalMargin) / 3, 
                       text = "B score: %d" % data.board.GetPlayerScore(0), anchor=SW)
    canvas.create_window(data.width - data.verticalMargin, data.horizontalMargin / 5,
                width = data.passButtonWidth, window = data.passButton, anchor=NE)
    canvas.create_window(data.width - data.verticalMargin, 4 * data.horizontalMargin / 5,
                width = data.evaluatorButtonWidth, window = data.modelEvaluateButton, anchor=NE)

    if data.gameOver:
        message = "GAME OVER"
        canvas.create_text(data.width / 2, data.horizontalMargin / 2, text=message)
    elif data.board.current_player in data.userPlayers:
        canvas.create_text(data.width / 2, data.horizontalMargin / 3, text="Your Turn")
        if data.board.current_player == 0:
            color = 'black'
        else:
            color = 'white'
        cellRadius = data.cellWidth / 2 - data.circleMargin
        cellRadius /= 2
        canvas.create_oval(data.width / 2 - cellRadius, data.horizontalMargin - cellRadius,
                           data.width / 2 + cellRadius, data.horizontalMargin + cellRadius, fill=color)

def run(width=300, height=300):
    def redrawAllWrapper(canvas, data):
        canvas.delete(ALL)
        canvas.create_rectangle(0, 0, data.width, data.height,
                                fill='white', width=0)
        redrawAll(canvas, data)
        canvas.update()
        data.require_redraw = False

    def mousePressedWrapper(event, canvas, data):
        mousePressed(event, data)
        redrawAllWrapper(canvas, data)

    def keyPressedWrapper(event, canvas, data):
        keyPressed(event, data)
        redrawAllWrapper(canvas, data)

    def timerFiredWrapper(canvas, data):
        timerFired(data)
        redrawAllWrapper(canvas, data)
        # pause, then call timerFired again
        canvas.after(data.timerDelay, timerFiredWrapper, canvas, data)
    # Set up data and call init
    class Struct(object): pass
    data = Struct()
    data.width = width
    data.height = height
    data.timerDelay = 100 # milliseconds

    # create the root and the canvas
    root = Tk()
    root.wm_title("Go")
    root.resizable(width=False, height=False) # prevents resizing window

    data.root = root
    canvas = Canvas(root, width=data.width, height=data.height)
    canvas.configure(bd=0, highlightthickness=0)
    canvas.pack()

    init(data)

    # set up events
    root.bind("<Button-1>", lambda event:
                            mousePressedWrapper(event, canvas, data))
    root.bind("<Key>", lambda event:
                            keyPressedWrapper(event, canvas, data))
    timerFiredWrapper(canvas, data)
    # and launch the app
    root.mainloop()  # blocks until window is closed
    print("bye!")

if __name__ == "__main__":
    run(1200, 1200)
