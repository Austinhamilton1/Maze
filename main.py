import argparse
import time

from maze import Maze

if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        prog='Maze Generator',
        description='Generate a maze of arbitrary size',
    )

    parser.add_argument('-r', '--rows', type=int)
    parser.add_argument('-c', '--cols', type=int)
    parser.add_argument('-g', '--gen-alg')
    parser.add_argument('-s', '--solve-alg')
    parser.add_argument('-o', '--output-file')
    parser.add_argument('-i', '--input-file')
    parser.add_argument('-p', '--print', action='store_true')

    args = parser.parse_args()

    rows = args.rows
    cols = args.cols
    gen_alg = args.gen_alg
    solve_alg = args.solve_alg
    output_file = args.output_file
    input_file = args.input_file
    print_maze = args.print

    if input_file != None:
        maze = Maze.from_file(input_file)
    else:
        maze = Maze(rows, cols)
        maze.generate_maze(gen_alg)
    
    if solve_alg != None:
        maze.solve(solve_alg)

    if print_maze:
        print(maze)

    if output_file != None:
        maze.to_file(output_file)