import json
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

class Graph:
    def __init__(self):
        self.graph = {}

    def add(self, f, t):
        if f not in self.graph:
            self.graph[f] = []
        self.graph[f].append(t)

def train_test_split(
        sessions: list[list[int]],
) -> tuple[list[list[int]], list[int]]:
    train_sessions = [session[:-1] for session in sessions]
    test_targets = [session[-1] for session in sessions]
    return train_sessions, test_targets

def get_size(
        sessions: list[list[int]]
) -> int:
    size = max(sessions[0])
    for i in range(1, len(sessions)):
        size = max(max(sessions[i]), size)
    return size

def get_probabilities(
        probabilties: list[list[float]],
        graph: dict,
        key: int
) -> None:
    frequency = Counter(graph[key])
    for num in frequency.keys():
        probabilties[key - 1][num - 1] += frequency[num] / len(graph[key])

def find_probabilities(
        probabilties: list[list[float]],
        graph: Graph
) -> None:
        for key in graph.graph.keys():
            get_probabilities(probabilties, graph.graph, key)

def get_top_ten(
        probabilities: list[list[float]],
        good_id: int
) -> list[int]:
    arr = []
    for i in range(len(probabilities[good_id])):
        arr.append((i, probabilities[good_id][i]))
    arr.sort(key=lambda x: x[1], reverse=True)
    return [arr[i][0] + 1 for i in range(10)]

def hit_at_k(
    recommendations: list[list[int]],
    true_items: list[int],
    k: int = 10,
) -> float:
    assert len(recommendations) == len(true_items)
    
    hits = 0
    for recs, true_item in zip(recommendations, true_items):
        if true_item in recs[:k]:
            hits += 1

    return hits / len(true_items)

def build_and_save_graphs(
        sessions: list[list[int]],
        output_dir: Path
) -> None:
    session_lengths = pd.Series([len(s) for s in sessions], name="session_length")
    all_items = pd.Series([item for s in sessions for item in s], name="item_id")
    item_freq = all_items.value_counts()

    plt.figure(figsize=(8, 5))
    session_lengths.plot.hist(bins=30, edgecolor="black")
    plt.title("Распределение длины сессий")
    plt.xlabel("Длина сессии")
    plt.ylabel("Количество сессий")
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    top20 = item_freq.head(20)
    plt.figure(figsize=(11, 5))
    top20.plot.bar()
    plt.title("Топ-20 самых частых товаров")
    plt.xlabel("ID товара")
    plt.ylabel("Частота")
    plt.xticks(rotation=60)
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

    sorted_freq = sorted(item_freq.values, reverse=True)
    plt.figure(figsize=(8, 5))
    plt.plot(range(1, len(sorted_freq) + 1), sorted_freq)
    plt.xscale("log")
    plt.yscale("log")
    plt.title("Частота товара по рангу")
    plt.xlabel("Ранг товара")
    plt.ylabel("Частота")
    plt.tight_layout()
    plt.show(block=True)
    plt.close()

def main():
    base_dir = Path(__file__).resolve().parent
    sessions = []
    with (base_dir / "sessions.jsonl").open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                plt.show()
                sessions.append(json.loads(line))
              
    build_and_save_graphs(sessions, base_dir / "analysis_plots")

    data = train_test_split(sessions)
    train = data[0]
    target = data[1]

    graph = Graph()
    for i in range(len(train)):
        session = train[i]
        for j in range(len(session) - 1):
            graph.add(session[j], session[j + 1])

    size = get_size(sessions)

    probabilities = [[0.0 for _ in range(size)] for _ in range(size)]
    find_probabilities(probabilities, graph)

    popular = [item for item, _ in Counter(item for session in train for item in session).most_common(10)]
    print(popular)

    recs = [get_top_ten(probabilities, train[i][-1] - 1) for i in range(len(sessions))]
    top10 = [popular] * len(sessions)

    print(f'Наша модель: {hit_at_k(recs, target):.0%}')
    print(f'10 самых популярных: {hit_at_k(top10, target):.0%}' )

if __name__ == "__main__":
    main()
