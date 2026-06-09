import numpy as np

class Env:
    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3

    def generate_tile(self):
        empty_cells = np.argwhere(self.board == 0)
        idx = np.random.choice(len(empty_cells))
        row, col = empty_cells[idx]
        self.board[row][col] = 2 if np.random.random() < 0.9 else 4

    def __generate_board(self):
        grid = np.zeros((4, 4), dtype=int)

        return grid

    def __init__(self):
        self.board = self.__generate_board()
        self.generate_tile()
        self.generate_tile()
        self.score = 0
        self.done = False



    def reset(self):
        self.board = np.zeros((4, 4), dtype=int)
        self.score = 0
        self.generate_tile()
        self.generate_tile()
        self.done = False


    def step(self, action):
        if action == self.LEFT:
            self.merge()
        elif action == self.RIGHT:
            self.board = np.rot90(self.board, 2)
            self.merge()
            self.board = np.rot90(self.board, 2)
        elif action == self.UP:
            self.board = np.rot90(self.board, 1)
            self.merge()
            self.board = np.rot90(self.board, 3)
        elif action == self.DOWN:
            self.board = np.rot90(self.board, 3)
            self.merge()
            self.board = np.rot90(self.board, 1)

        if np.all(self.board != 0):
            self.done = True
        elif not any(self.valid(a) for a in [self.UP, self.DOWN, self.LEFT, self.RIGHT]):
            self.done = True
        else:
            self.generate_tile()
        return self.board,self.score,self.done

    def merge(self):
        for i in range(4):
            row = self.board[i]
            tiles = row[row != 0].tolist()
            j = 0
            length = len(tiles)

            while j < length - 1 :
                if tiles[j] == tiles[j + 1]:
                    val = tiles.pop(j)
                    tiles.pop(j)
                    tiles.insert(j, 2*val)
                    self.score += 2*val
                    length -= 1
                j += 1

            tiles += [0] * (4 - len(tiles))
            self.board[i] = tiles

    def valid(self,action):
        board_copy = self.board.copy()
        if action == self.LEFT:
            pass
        elif action == self.RIGHT:
            board_copy = np.rot90(self.board, 2)
        elif action == self.UP:
            board_copy = np.rot90(self.board, 1)
        elif action == self.DOWN:
            board_copy = np.rot90(self.board, 3)

        for i in range(4):
            row = board_copy[i]
            for j in range(3):
                if board_copy[i][j] == board_copy[i][j + 1]:
                    return True
                elif board_copy[i][j] == 0:
                    return True

        return False



