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
    parser.add_argument('-p', '--print', action='store_true')

    args = parser.parse_args()

    rows = args.rows
    cols = args.cols
    gen_alg = args.gen_alg
    solve_alg = args.solve_alg
    output_file = args.output_file
    print_maze = args.print

    maze = Maze(rows, cols)
    generate_start = time.time()
    maze.generate_maze(gen_alg)
    generate_end = time.time()
    solve_start = time.time()
    maze.solve(solve_alg)
    solve_end = time.time()
    if print_maze:
        print(maze)
    print(f'Maze generated in {generate_end - generate_start} seconds')
    print(f'Maze solved in {solve_end - solve_start} seconds')

    if output_file != None:
        with open(output_file, 'w+') as file:
            output_file.write(str(maze))