
import ctypes
from typing import *

LIB = None


def LoadLib():
    board_lib_path = "bazel-bin/code/cpp_mcts/libboard.so"
    global LIB
    LIB = ctypes.cdll.LoadLibrary(board_lib_path)

    LIB.NewBoard.argtypes = [ctypes.c_int]
    LIB.NewBoard.restype = ctypes.c_void_p

    LIB.BoardSize.argtypes = [ctypes.c_void_p]
    LIB.BoardSize.restype = ctypes.c_int

    LIB.LegalMove.argtypes = [ctypes.c_void_p, ctypes.c_int, ctypes.c_int]
    LIB.LegalMove.restype = ctypes.c_bool

    LIB.StoneList.argtypes = [ctypes.c_void_p]
    LIB.StoneList.restype = StoneList

    LIB.MakeMove.argtypes = [ctypes.c_void_p,
                             ctypes.c_int, ctypes.c_int, ctypes.c_int]

    LIB.PlayerScore.argtypes = [ctypes.c_void_p, ctypes.c_int]
    LIB.PlayerScore.restype = ctypes.c_int

    LIB.GetMCTSMove.argtypes = [ctypes.c_void_p,
                                ctypes.c_int, ctypes.c_int, ctypes.c_int]
    LIB.GetMCTSMove.restype = MCTSMove

    LIB.LegalMoves.argtypes = [ctypes.c_void_p]
    LIB.LegalMoves.restype = LocationList

    LIB.GameOver.argtypes = [ctypes.c_void_p]
    LIB.GameOver.restype = ctypes.c_bool

    LIB.GetWinner.argtypes = [ctypes.c_void_p]
    LIB.GetWinner.restype = ctypes.c_int

    LIB.Copy.argtypes = [ctypes.c_void_p]
    LIB.Copy.restype = ctypes.c_void_p

    LIB.RandomPlayout.argtypes = [ctypes.c_void_p, ctypes.c_int]
    LIB.RandomPlayout.restype = ctypes.c_int

    LIB.GameEffectivelyOver.argtypes = [ctypes.c_void_p]
    LIB.GameEffectivelyOver.restype = ctypes.c_bool

    LIB.PassWins.argtypes = [ctypes.c_void_p, ctypes.c_int]
    LIB.PassWins.restype = ctypes.c_bool


class Policy(ctypes.Structure):
    _fields_ = [('length', ctypes.c_int),
                ('distribution', ctypes.POINTER(ctypes.c_float))]


class MCTSMove(ctypes.Structure):
    _fields_ = [('row', ctypes.c_int), ('col', ctypes.c_int),
                ('confidence', ctypes.c_double), ('policy', Policy)]


class Stone(ctypes.Structure):
    _fields_ = [('row', ctypes.c_int), ('col', ctypes.c_int),
                ('stone_type', ctypes.c_char)]


class StoneList(ctypes.Structure):
    _fields_ = [('length', ctypes.c_int),
                ('stones', ctypes.POINTER(Stone))]


class Location(ctypes.Structure):
    _fields_ = [('row', ctypes.c_int), ('col', ctypes.c_int)]


class LocationList(ctypes.Structure):
    _fields_ = [('length', ctypes.c_int),
                ('locations', ctypes.POINTER(Location))]


class Board():
    # Give a positive board size to initiate an empty Board. A non-positive
    # board size is used for the copy constructor
    def __init__(self, board_size: int):
        if LIB is None:
            LoadLib()
        if board_size > 0:
            self.c_board = LIB.NewBoard(board_size)
            self.current_player = 0
            self.size = board_size

    def __len__(self):
        return LIB.BoardSize(self.c_board)

    def __repr__(self):
        board_list = self.BoardList()
        result = ''
        for row in board_list:
            result += str(row) + '\n'
        return result

    def Copy(self):
        copy = Board(-1)
        copy.c_board = LIB.Copy(self.c_board)
        copy.current_player = self.current_player
        copy.size = self.size
        return copy

    def LegalMove(self, row: int, col: int) -> bool:
        return LIB.LegalMove(self.c_board, row, col)

    def BoardList(self) -> List[List[str]]:
        stone_list = LIB.StoneList(self.c_board)
        board_list = [['-'] * self.size for _ in range(self.size)]
        for i in range(stone_list.length):
            stone = stone_list.stones[i]
            board_list[stone.row][stone.col] = stone.stone_type.decode()
        return board_list

    def MakeMove(self, row: int, col: int):
        LIB.MakeMove(self.c_board, row, col, self.current_player)
        self.current_player = 1 - self.current_player

    def GetPlayerScore(self, player: int) -> int:
        return LIB.PlayerScore(self.c_board, player)

    def GetMCTSMove(self, threads: int, seconds: int) -> MCTSMove:
        return LIB.GetMCTSMove(self.c_board, threads, seconds * 1000, self.current_player)

    def GetLegalMoves(self) -> List[Location]:
        move_list = LIB.LegalMoves(self.c_board)
        moves = [move_list.locations[i] for i in range(move_list.length)]
        return moves

    def IsGameOver(self) -> bool:
        return LIB.GameOver(self.c_board)

    def GetWinner(self) -> int:
        return LIB.GetWinner(self.c_board)

    def RandomPlayout(self) -> int:
        return LIB.RandomPlayout(self.c_board, self.current_player)

    def GameEffectivelyOver(self) -> bool:
        return LIB.GameEffectivelyOver(self.c_board)

    def PassWins(self) -> bool:
        return LIB.PassWins(self.c_board, self.current_player)


if __name__ == "__main__":
    LoadLib()

    size = 13
    b = Board(size)
    print(len(b))
    print(b)

    b.MakeMove(1, 1)
    print(b)

    b.MakeMove(2, 2)
    print(b)
