import numpy as np
from collections import deque
import pickle
    
class Maze:
    #these are the bits in the cells array
    NORTH = 1
    SOUTH = 2
    EAST = 4
    WEST = 8
    GEN_VISIT = 16
    SOL_PATH = 32
    SOL_VISIT = 64
    FREE_USE_BIT = 128

    def __init__(self, rows, cols):
        self.rows = rows
        self.cols = cols

        #entrance and exit of the maze
        self.top = np.random.randint(self.cols)
        self.bottom = np.random.randint(self.cols)

        #holds data about the cells
        #bits 1-4 hold info on walls (0 = wall, 1 = no wall)
        #bit 5 determines if the cell has been visited during generation
        #bit 6 determines if the cell is on the solution path
        #bit 7 determines if the cell has been visited during solving
        #bit 8 is left as a free bit for the user (e.g., cell is occupied by player)
        self.cells = np.full((self.rows, self.cols), 0, dtype=np.uint8)

    def get_unvisited_neighbors(self, row, col, solving):
        '''
        Given a row and column, returns a list of all the cells
        neighboring the cell at (row, col) that haven't been visited
        
        Parameters:
        -----------
        row: int
            - The row to check
        col: int
            - The column to check
        solving: bool
            - True if solving, False if generating

        Returns:
        --------
        list[tuple(int, int)]
            - A list of neighboring indexes that are unvisited
        '''
        neighbors = []

        if row != 0 and not self._is_visited(row-1, col, solving):
            if not solving or solving and (self.cells[row,col] & 1) == 1:
                neighbors.append((row-1, col))
        if row != self.rows-1 and not self._is_visited(row+1, col, solving):
            if not solving or solving and (self.cells[row,col] & 2) >> 1 == 1:
                neighbors.append((row+1, col))
        if col != self.cols-1 and not self._is_visited(row, col+1, solving):
            if not solving or solving and (self.cells[row,col] & 4) >> 2 == 1:
                neighbors.append((row, col+1))
        if col != 0 and not self._is_visited(row, col-1, solving):
            if not solving or solving and (self.cells[row,col] & 8) >> 3 == 1:
                neighbors.append((row, col-1))

        return neighbors
    
    def get_neighbors(self, row, col):
        '''
        Given a row and column, returns a list of the cells neighbors

        Parameters:
        -----------
        row: int
            - The row to check
        col: int
            - The column to check
        
        Returns:
        --------
        list[tuple(int, int)]
            - A list of neighboring indexes
        '''

        neighbors = []
        if row != 0:
            neighbors.append((row-1, col))
        if row != self.rows-1:
            neighbors.append((row+1, col))
        if col != 0:
            neighbors.append((row, col-1))
        if col != self.cols-1:
            neighbors.append((row, col+1))
        return neighbors
    
    def _is_visited(self, row, col, solving):
        '''
        Determines if a cell has been visited based on the value set in the 5th
        bit of the cell array at a given index

        Parameters:
        -----------
        row: int
            - The row to check
        col: int
            - The column to check
        solving: bool
            - True if solving, False if generating
        
        Returns:
        --------
        True if the cell at index (row, col) has been visited, False otherwise
        '''
        mask = Maze.SOL_VISIT if solving else Maze.GEN_VISIT
        shift = 6 if solving else 4
        return np.right_shift((np.bitwise_and(self.cells[row,col], mask)), shift) == 1
    
    def generate_maze(self, algorithm='classic'):
        '''
        Generates a random maze

        Parameters:
        -----------
        Algorithm: str | None
            - The algorithm to generate the maze with (None = DFS)
        '''
        table = {
            None: self._classic_generate,
            'classic': self._classic_generate,
            'rooms': self._rooms_generate,
            'quick': self._quick_generate,
            'uniform': self._uniform_generate
        }
        if algorithm not in table:
            raise ValueError(f'Error: {algorithm} - Not supported')
        table[algorithm]()

    def _classic_generate(self):
        '''
        Generate a maze using a randomized depth first search algorithm
        '''
        #start the algorithm at a random cell at the top of the maze
        start = (0, self.top)
        self.cells[start] = np.bitwise_or(self.cells[start], Maze.GEN_VISIT)

        #used to facilitate DFS
        stack = [start]

        #Randomized DFS
        while len(stack) > 0:
            current = stack[-1]
            neighborhood = self.get_unvisited_neighbors(current[0], current[1], False)
            if len(neighborhood) > 0:
                #if the current cell has unvisisted neighbors
                #connect to a random neighbor
                next_cell = neighborhood[np.random.randint(len(neighborhood))]
                if next_cell[0] < current[0]:
                    #next cell is north of current
                    self.cells[current] = np.bitwise_or(self.cells[current], Maze.NORTH)
                    self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.SOUTH)
                elif next_cell[0] > current[0]:
                    #next cell is south of current
                    self.cells[current] = np.bitwise_or(self.cells[current], Maze.SOUTH)
                    self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.NORTH)
                elif next_cell[1] > current[1]:
                    #next cell is east of current
                    self.cells[current] = np.bitwise_or(self.cells[current], Maze.EAST)
                    self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.WEST)
                elif next_cell[1] < current[1]:
                    #next cell is west of current
                    self.cells[current] = np.bitwise_or(self.cells[current], Maze.WEST)
                    self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.EAST)
                
                #mark the cell visited and add it to the stack
                self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.GEN_VISIT)
                stack.append(next_cell)
            else:
                #otherwise remove the cell from the search
                stack.pop()

    def _rooms_generate(self):
        '''
        Generate a maze using an algorithm that generates large rooms
        '''
        #start the algorithm at a random cell at the top of the maze
        start = (0, self.top)
        
        #used to facilitate the algorithm
        frontier = self.get_unvisited_neighbors(start[0], start[1], False)
        maze = {start}

        #mark the first cell as visited
        self.cells[start] = np.bitwise_or(self.cells[start], Maze.GEN_VISIT)

        #generation algorithm
        while len(frontier) != 0:
            #get a random node from the frontier
            index = np.random.randint(len(frontier))
            next_cell = frontier[index]

            #connect the new node to one of the nodes already in the maze
            current = (-1, -1)
            neighbors = self.get_neighbors(next_cell[0], next_cell[1])
            np.random.shuffle(neighbors)
            for neighbor in neighbors:
                if neighbor in maze:
                    current = neighbor
                    break
            if next_cell[0] < current[0]:
                #next cell is north of current
                self.cells[current] = np.bitwise_or(self.cells[current], Maze.NORTH)
                self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.SOUTH)
            elif next_cell[0] > current[0]:
                #next cell is south of current
                self.cells[current] = np.bitwise_or(self.cells[current], Maze.SOUTH)
                self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.NORTH)
            elif next_cell[1] > current[1]:
                #next cell is east of current
                self.cells[current] = np.bitwise_or(self.cells[current], Maze.EAST)
                self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.WEST)
            elif next_cell[1] < current[1]:
                #next cell is west of current
                self.cells[current] = np.bitwise_or(self.cells[current], Maze.WEST)
                self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.EAST)
            
            #mark the cell as visited and remove it from the frontier
            self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.GEN_VISIT)
            del frontier[index]
            maze.add(next_cell)

            #add the cells neighbors to the frontier
            for neighbor in self.get_unvisited_neighbors(next_cell[0], next_cell[1], False):
                frontier.append(neighbor)

    def _quick_generate(self):
        '''
        Quickly generate a maze using a tesselation algorithm
        '''
        #initalize the cells to a single matrix
        self.cells = np.full((1,1), 0, dtype=np.uint8)
        rows = 1
        cols = 1

        #rows and columns double each round, so go until both are larger than the input
        while rows < self.rows and cols < self.cols:
            #copy the cells three times in a tesselation to make a maze
            #twice as big
            new_cells = np.zeros((self.cells.shape[0]*2, self.cells.shape[1]*2), dtype=np.uint8)
            for i in range(len(new_cells)):
                for j in range(len(new_cells)):
                    new_cells[i,j] = self.cells[i % len(self.cells), j % len(self.cells)]
            
            #we need to connect the three "rooms"
            opening1 = (rows-1, np.random.randint(cols))
            opening2 = (np.random.randint(rows, len(new_cells)), cols-1)
            opening3 = (rows-1, np.random.randint(cols, len(new_cells)))
            
            #connect the top left room to the bottom left room
            new_cells[opening1] = np.bitwise_or(new_cells[opening1], Maze.SOUTH)
            new_cells[opening1[0]+1, opening1[1]] = np.bitwise_or(new_cells[opening1[0]+1, opening1[1]], Maze.NORTH)

            #connect the bottom left room to the bottom right room
            new_cells[opening2] = np.bitwise_or(new_cells[opening2], Maze.EAST)
            new_cells[opening2[0], opening2[1]+1] = np.bitwise_or(new_cells[opening2[0], opening2[1]+1], Maze.WEST)

            #connect the top right room to the bottom right room
            new_cells[opening3] = np.bitwise_or(new_cells[opening3], Maze.SOUTH)
            new_cells[opening3[0]+1, opening3[1]] = np.bitwise_or(new_cells[opening3[0]+1, opening3[1]], Maze.NORTH)

            #update our cells
            self.cells = new_cells
            rows *= 2
            cols *= 2

        #update the rows and columns of the maze
        self.rows = rows
        self.cols = cols

    def _uniform_generate(self):
        '''
        Generate a uniform spanning maze using a loop-erased random walk
        '''
        #ininitalize the maze
        unvisited = set([(i, j) for i in range(self.rows) for j in range(self.cols)])
        start = list(unvisited)[np.random.randint(len(unvisited))]
        unvisited.remove(start)

        #loop-erased random walk
        while len(unvisited) > 0:
            #start with a random cell and add it to the path
            cell = list(unvisited)[np.random.randint(len(unvisited))]
            path = [cell]

            #loop until the path connects to the maze
            while cell in unvisited:
                neighbors = self.get_neighbors(cell[0], cell[1])
                cell = neighbors[np.random.randint(len(neighbors))]
                if cell in path:
                    #remove loops
                    path = path[:path.index(cell)+1]
                else:
                    path.append(cell)

            #link the cells in the path
            for i in range(len(path)-1):
                current = path[i]
                next_cell = path[i+1]

                if next_cell[0] < current[0]:
                    #next cell is north of current
                    self.cells[current] = np.bitwise_or(self.cells[current], Maze.NORTH)
                    self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.SOUTH)
                elif next_cell[0] > current[0]:
                    #next cell is south of current
                    self.cells[current] = np.bitwise_or(self.cells[current], Maze.SOUTH)
                    self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.NORTH)
                elif next_cell[1] > current[1]:
                    #next cell is east of current
                    self.cells[current] = np.bitwise_or(self.cells[current], Maze.EAST)
                    self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.WEST)
                elif next_cell[1] < current[1]:
                    #next cell is west of current
                    self.cells[current] = np.bitwise_or(self.cells[current], Maze.WEST)
                    self.cells[next_cell] = np.bitwise_or(self.cells[next_cell], Maze.EAST)

                #update the unvisited set
                unvisited.remove(path[i])


    def solve(self, algorithm='dfs'):
        '''
        Solves the maze

        Parameters:
        -----------
        Algorithm: str | None
            - The algorithm to solve the maze with (None = DFS)
        '''
        table = {
            None: self._dfs_solve,
            'dfs': self._dfs_solve,
            'bfs': self._bfs_solve
        }
        if algorithm not in table:
            raise ValueError(f'Error: {algorithm} - Unsupported')
        table[algorithm]()

    def _dfs_solve(self):
        '''
        Solve the maze using a depth first search
        '''
        #start the algorithm at the top of the maze
        start = (0, self.top)
        #visited
        self.cells[start] = np.bitwise_or(self.cells[start], Maze.SOL_VISIT)

        #used to facilitate DFS
        stack = [start]

        #DFS
        while len(stack) > 0:
            current = stack[-1]
            #assume current is on solution path
            self.cells[current] = np.bitwise_or(self.cells[current], Maze.SOL_PATH)
            if current == (self.rows-1, self.bottom):
                break
            neighborhood = self.get_unvisited_neighbors(current[0], current[1], True)
            if len(neighborhood) > 0:
                #add the neighbors to the stack
                for neighbor in neighborhood:
                    #mark visited
                    self.cells[neighbor] = np.bitwise_or(self.cells[neighbor], Maze.SOL_VISIT)
                    stack.append(neighbor)
            else:
                #pop the stack and set the current cell so its not on the solution path
                stack.pop()
                self.cells[current] = np.bitwise_and(self.cells[current], np.bitwise_xor(0xff, Maze.SOL_PATH))

    def _bfs_solve(self):
        '''
        Solve the maze using a breadth first search
        '''
        #start the algorithm at the top of the maze
        start = (0, self.top)
        #visited
        self.cells[start] = np.bitwise_or(self.cells[start], Maze.SOL_VISIT)
        #used to find the path once the graph has been traversed
        parents = {}

        #used to facilitate BFS
        queue = deque([start])

        #BFS
        while len(queue) > 0:
            current = queue.popleft()
            if current == (self.rows-1, self.bottom):
                break
            neighborhood = self.get_unvisited_neighbors(current[0], current[1], True)
            if len(neighborhood) > 0:
                #add neighbors to the queue
                for neighbor in neighborhood:
                    #mark visited
                    self.cells[neighbor] = np.bitwise_or(self.cells[neighbor], Maze.SOL_VISIT)
                    queue.append(neighbor)
                    parents[neighbor] = current
        
        #BFS backtrace
        current = (self.rows-1, self.bottom)
        while current != (0, self.top):
            self.cells[current] = np.bitwise_or(self.cells[current], Maze.SOL_PATH)
            current = parents[current]
        self.cells[0, self.top] = np.bitwise_or(self.cells[0, self.top], Maze.SOL_PATH)

    def __str__(self):
        '''
        Converts a maze to a string representation
        '''
        #top row
        string = ' '
        for i in range(self.cols):
            if i != self.top:
                string += '_ '
            else:
                string += '  '
        string += '\n'
        #maze
        for i in range(self.rows):
            #left side
            string += '|'
            for j in range(self.cols):
                #exit point
                if i == self.rows-1 and j == self.bottom:
                    if (self.cells[i,j] & Maze.EAST) >> 2 == 0:
                        string += '*|'
                    else:
                        string += '* '
                else:
                    if (self.cells[i,j] & Maze.SOUTH) >> 1 == 0:
                        #has a south wall
                        if (self.cells[i,j] & Maze.SOL_PATH) >> 5 == 1:
                            #on solution path
                            string += '\033[4m' + '*' + '\033[0m'
                        else:
                            #not on solution path
                            string += '_'
                    else:
                        #no south wall
                        if (self.cells[i,j] & Maze.SOL_PATH) >> 5 == 1:
                            #on solution path
                            string += '*'
                        else:
                            #not on solution path
                            string += ' '
                    if (self.cells[i,j] & Maze.EAST) >> 2 == 0:
                        #has an east wall
                        string += '|'
                    else:
                        #no east wall
                        string += ' '
            string += '\n'
        return string
    
    def to_file(self, file_name):
        with open(file_name, 'wb+') as file:
            pickle.dump(self, file)

    @staticmethod
    def from_file(file_name):
        with open(file_name, 'rb') as file:
            return pickle.load(file)