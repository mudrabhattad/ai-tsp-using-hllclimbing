import tkinter as tk
from tkinter import ttk
from tkinter import messagebox
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class TSPSolverApp:
    def __init__(self, root):
        self.root = root
        self.root.title("TSP Solver")
        self.root.geometry("400x300")
        
        self.source_city_var = tk.StringVar()
        self.result_text = tk.StringVar()
        
        self.init_widgets()

    def init_widgets(self):
        input_frame = ttk.Frame(self.root, padding=(20, 10))
        input_frame.pack(fill=tk.BOTH, expand=True)
        
        ttk.Label(input_frame, text="Source City:").grid(row=0, column=0, sticky="w")
        self.source_city_entry = ttk.Entry(input_frame, textvariable=self.source_city_var)
        self.source_city_entry.grid(row=0, column=1, padx=5, pady=5, sticky="we")

        self.solve_button = ttk.Button(input_frame, text="Solve TSP", command=self.solve_tsp)
        self.solve_button.grid(row=1, column=1, pady=10)

        self.result_label = ttk.Label(input_frame, textvariable=self.result_text)
        self.result_label.grid(row=2, column=0, columnspan=2)

    def solve_tsp(self):
        source_city = self.source_city_var.get()
        if source_city:
            try:
                df = pd.read_csv("indianct1.csv")
                df.dropna(inplace=True)
                df['city'] = df['city'].str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8')

                df = df.head(25).reset_index(drop=True)

                distances = np.zeros((len(df), len(df)))
                for i in range(len(df)):
                    for j in range(len(df)):
                        distances[i][j] = self.calculate_distance(df['lat'][i], df['lng'][i], df['lat'][j], df['lng'][j])

                source_city_index = df[df['city'] == source_city].index[0]

                initial_solution = list(range(len(df)))
                initial_solution.remove(source_city_index)
                initial_solution.insert(0, source_city_index)

                optimized_route, total_distance = self.tsp_hill_climbing(distances, initial_solution, max_iterations=10000)

                optimized_route.remove(source_city_index)
                optimized_route.insert(0, source_city_index)

                self.plot_route(df, optimized_route, distances)

                result_str = "Optimized route:\n"
                for i, city_idx in enumerate(optimized_route):
                    result_str += f"{i+1}: {df['city'][city_idx]}\n"
                result_str += f"Total distance traveled: {total_distance} km"

                self.result_text.set(result_str)

            except Exception as e:
                messagebox.showerror("Error", f"Failed to solve TSP: {str(e)}")
        else:
            messagebox.showerror("Error", "Please enter a source city.")

    def calculate_distance(self, lat1, lon1, lat2, lon2):
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        r = 6371
        return c * r

    def objective_function(self, route, distances):
        total_distance = 0
        for i in range(len(route) - 1):
            total_distance += distances[route[i]][route[i+1]]
        total_distance += distances[route[-1]][route[0]]
        return total_distance

    def tsp_hill_climbing(self, distances, initial_solution, max_iterations):
        current_solution = initial_solution
        current_distance = self.objective_function(current_solution, distances)

        for _ in range(max_iterations):
            neighbor_solution = current_solution.copy()
            idx1, idx2 = np.random.choice(len(neighbor_solution), 2, replace=False)
            neighbor_solution[idx1], neighbor_solution[idx2] = neighbor_solution[idx2], neighbor_solution[idx1]

            neighbor_distance = self.objective_function(neighbor_solution, distances)

            if neighbor_distance < current_distance:
                current_solution = neighbor_solution
                current_distance = neighbor_distance

        return current_solution, current_distance

    def plot_route(self, df, optimized_route, distances):
        plt.figure(figsize=(10, 6))
        for i, city_idx in enumerate(optimized_route):
            plt.text(df['lng'][city_idx], df['lat'][city_idx], f"{i+1}: {df['city'][city_idx]}", fontsize=8)

        plt.plot(df['lng'][optimized_route + [optimized_route[0]]],
                 df['lat'][optimized_route + [optimized_route[0]]],
                 c='red', linewidth=2)

        for i in range(len(optimized_route) - 1):
            city1 = optimized_route[i]
            city2 = optimized_route[i + 1]
            weight = distances[city1][city2]
            plt.text((df['lng'][city1] + df['lng'][city2]) / 2,
                     (df['lat'][city1] + df['lat'][city2]) / 2,
                     f"{weight:.2f} km", fontsize=6)

        plt.xlabel('Longitude')
        plt.ylabel('Latitude')
        plt.title('Optimized Route')
        plt.grid(True)
        plt.show()

def main():
    root = tk.Tk()
    app = TSPSolverApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()
