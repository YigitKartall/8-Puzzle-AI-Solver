import tkinter as tk
from tkinter import messagebox, filedialog
import numpy as np
import time
import heapq
import threading
from collections import deque


class PuzzleState:
    def __init__(self, board, parent=None, move="", depth=0, cost=0):
        self.board = np.array(board)
        self.parent = parent
        self.move = move
        self.depth = depth
        self.cost = cost
        self.size = int(len(board) ** 0.5)

    def __lt__(self, other):
        return self.cost < other.cost

    def successors(self):
        nodes = []
        idx = np.where(self.board == 0)[0][0]
        r, c = divmod(idx, self.size)
        moves = {"Up": (-1, 0), "Down": (1, 0), "Left": (0, -1), "Right": (0, 1)}

        for move_name, (dr, dc) in moves.items():
            nr, nc = r + dr, c + dc
            if 0 <= nr < self.size and 0 <= nc < self.size:
                new_board = self.board.copy()
                new_board[r * self.size + c], new_board[nr * self.size + nc] = \
                    new_board[nr * self.size + nc], new_board[r * self.size + c]
                nodes.append(PuzzleState(new_board, self, move_name, self.depth + 1))
        return nodes


class SearchAlgorithms:
    @staticmethod
    def bfs(start_state, goal_board, update_cb):
        queue = deque([start_state])
        visited = {tuple(start_state.board)}
        expanded = 0
        while queue:
            current = queue.popleft()
            expanded += 1
            if expanded % 100 == 0: update_cb(expanded)
            if np.array_equal(current.board, goal_board):
                return current, expanded
            for next_node in current.successors():
                if tuple(next_node.board) not in visited:
                    visited.add(tuple(next_node.board))
                    queue.append(next_node)
        return None, expanded

    @staticmethod
    def ids(start_state, goal_board, update_cb, max_depth=50):
        for limit in range(max_depth):
            result, expanded = SearchAlgorithms.dls(start_state, goal_board, limit, update_cb)
            if result: return result, expanded
        return None, 0

    @staticmethod
    def dls(state, goal, limit, update_cb):
        stack = [(state, {tuple(state.board)})]
        expanded = 0
        while stack:
            curr, path = stack.pop()
            expanded += 1
            if expanded % 50 == 0: update_cb(expanded)
            if np.array_equal(curr.board, goal): return curr, expanded
            if curr.depth < limit:
                for next_node in curr.successors():
                    if tuple(next_node.board) not in path:
                        new_path = path.copy()
                        new_path.add(tuple(next_node.board))
                        stack.append((next_node, new_path))
        return None, expanded

    @staticmethod
    def a_star(start_state, goal_board, h_type, update_cb):
        open_list = []
        heapq.heappush(open_list, start_state)
        visited = {tuple(start_state.board): start_state.depth}
        expanded = 0
        while open_list:
            current = heapq.heappop(open_list)
            expanded += 1
            if expanded % 50 == 0: update_cb(expanded)
            if np.array_equal(current.board, goal_board):
                return current, expanded
            for next_node in current.successors():
                h = SearchAlgorithms.manhattan(next_node.board, goal_board) if h_type == "manhattan" \
                    else SearchAlgorithms.misplaced(next_node.board, goal_board)
                next_node.cost = next_node.depth + h
                state_tup = tuple(next_node.board)
                if state_tup not in visited or next_node.depth < visited[state_tup]:
                    visited[state_tup] = next_node.depth
                    heapq.heappush(open_list, next_node)
        return None, expanded

    @staticmethod
    def manhattan(board, goal):
        dist = 0
        s = int(len(board) ** 0.5)
        for i in range(1, len(board)):
            curr = np.where(board == i)[0][0]
            target = np.where(goal == i)[0][0]
            dist += abs(curr // s - target // s) + abs(curr % s - target % s)
        return dist

    @staticmethod
    def misplaced(board, goal):
        return np.sum((board != goal) & (board != 0))


class PuzzleGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("8-Puzzle AI Solver")
        self.L = 3
        self.goal = np.append(np.arange(1, self.L * self.L), 0)
        self.current_board = self.goal.copy()

        self.canvas = tk.Canvas(root, width=300, height=300, bg="white")
        self.canvas.pack(pady=10)

        self.status_label = tk.Label(root, text="Nodes: 0 | Time: 0s", font=("Arial", 10, "bold"))
        self.status_label.pack()

        self.algo_var = tk.StringVar(value="A* Manhattan")
        tk.OptionMenu(root, self.algo_var, "BFS", "IDS", "A* Manhattan", "A* Misplaced").pack(pady=5)

        btn_frame = tk.Frame(root)
        btn_frame.pack(pady=10)
        tk.Button(btn_frame, text="Random", command=self.generate_random).grid(row=0, column=0, padx=5)
        tk.Button(btn_frame, text="Load File", command=self.load_file).grid(row=0, column=1, padx=5)
        tk.Button(btn_frame, text="Solve", command=self.run_solve_thread, bg="lightgreen").grid(row=0, column=2, padx=5)

        self.draw_board()

    def is_solvable(self, board):
        arr = board[board != 0]
        inv = sum(arr[i] > arr[j] for i in range(len(arr)) for j in range(i + 1, len(arr)))
        return inv % 2 == 0

    def draw_board(self):
        self.canvas.delete("all")
        cell = 300 // self.L
        for i, val in enumerate(self.current_board):
            r, c = divmod(i, self.L)
            if val != 0:
                self.canvas.create_rectangle(c * cell, r * cell, (c + 1) * cell, (r + 1) * cell, fill="skyblue")
                self.canvas.create_text(c * cell + cell / 2, r * cell + cell / 2, text=str(val), font=("Arial", 20))

    def generate_random(self):
        while True:
            np.random.shuffle(self.current_board)
            if self.is_solvable(self.current_board): break
        self.draw_board()

    def load_file(self):
        path = filedialog.askopenfilename()
        if path:
            with open(path, 'r') as f:
                content = f.read().replace('\n', ',').replace(' ', ',').split(',')
                self.current_board = np.array([int(x) for x in content if x.strip() != ""])
                self.draw_board()

    def run_solve_thread(self):
        threading.Thread(target=self.solve, daemon=True).start()

    def solve(self):
        start_state = PuzzleState(self.current_board)
        algo = self.algo_var.get()
        start_t = time.time()

        if algo == "BFS":
            res, nodes = SearchAlgorithms.bfs(start_state, self.goal, self.update_info)
        elif algo == "IDS":
            res, nodes = SearchAlgorithms.ids(start_state, self.goal, self.update_info)
        else:
            h = "manhattan" if "Manhattan" in algo else "misplaced"
            res, nodes = SearchAlgorithms.a_star(start_state, self.goal, h, self.update_info)

        elapsed = time.time() - start_t
        if res:
            path = []
            curr = res
            while curr:
                path.append(curr.board)
                curr = curr.parent
            self.animate_path(path[::-1], nodes, elapsed)
        else:
            messagebox.showinfo("Error", "Solution not found!")

    def update_info(self, expanded):
        self.status_label.config(text=f"Searching... Nodes: {expanded}")

    def animate_path(self, path, nodes, elapsed):
        for b in path:
            self.current_board = b
            self.draw_board()
            self.root.update()
            time.sleep(0.2)
        self.status_label.config(text=f"Nodes: {nodes} | Moves: {len(path) - 1} | Time: {elapsed:.2f}s")


if __name__ == "__main__":
    root = tk.Tk()
    app = PuzzleGUI(root)
    root.mainloop()
