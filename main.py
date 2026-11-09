"""
Лабораторная работа №2.
Вариант 4. Формирование набора функций программного продукта.
Студент: Якимов Максим Андреевич
Группа: Б24-517, номер в группе 30.

Запуск:
    python main.py
Результаты:
    - lab2_out/lab2_results.csv
    - lab2_out/lab2_convergence.png
"""

import os
import numpy as np
import matplotlib.pyplot as plt
import csv
from dataclasses import dataclass
from typing import List, Tuple

# ---------- Настройки воспроизводимости ----------
BASE_SEED = 42
N_RUNS = 20
N_FEATURES = 40  # Не менее 30 объектов по методичке

# ---------- Генерация данных ----------
@dataclass
class Feature:
    id: int
    cost: int
    time: int
    value: int
    depends_on: List[int]

def generate_data(seed: int) -> Tuple[List[Feature], int, int]:
    rng = np.random.default_rng(seed)
    features = []
    
    for i in range(N_FEATURES):
        cost = int(rng.integers(10, 50))
        time = int(rng.integers(1, 10))
        value = int(rng.integers(10, 100))
        
        depends_on = []
        if i > 0 and rng.random() < 0.3:
            dep_id = int(rng.integers(0, i))
            depends_on.append(dep_id)
            
        features.append(Feature(i, cost, time, value, depends_on))
        
    total_cost = sum(f.cost for f in features)
    total_time = sum(f.time for f in features)
    budget = int(total_cost * 0.4)
    deadline = int(total_time * 0.4)
    
    return features, budget, deadline

# ---------- Целевая функция и Ограничения ----------
def evaluate_fitness(solution: np.ndarray, features: List[Feature], budget: int, deadline: int, method: str) -> float:
    total_cost = 0
    total_time = 0
    total_value = 0
    dep_violations = 0
    
    for i, selected in enumerate(solution):
        if selected == 1:
            f = features[i]
            total_cost += f.cost
            total_time += f.time
            total_value += f.value
            for dep in f.depends_on:
                if solution[dep] == 0:
                    dep_violations += 1
                    
    if method == "penalty":
        penalty = 0
        if total_cost > budget:
            penalty += (total_cost - budget) * 100
        if total_time > deadline:
            penalty += (total_time - deadline) * 100
        penalty += dep_violations * 1000
        return -total_value + penalty
    else:
        return -total_value + dep_violations * 1000

def repair_solution(solution: np.ndarray, features: List[Feature], budget: int, deadline: int) -> np.ndarray:
    sol = solution.copy()
    n = len(features)

    changed = True
    while changed:
        changed = False
        for i in range(n):
            if sol[i] == 1:
                for dep in features[i].depends_on:
                    if sol[dep] == 0:
                        sol[dep] = 1
                        changed = True

    while True:
        total_cost = sum(features[i].cost for i in range(n) if sol[i] == 1)
        total_time = sum(features[i].time for i in range(n) if sol[i] == 1)

        if total_cost <= budget and total_time <= deadline:
            break

        is_dependency = np.zeros(n, dtype=bool)
        for i in range(n):
            if sol[i] == 1:
                for dep in features[i].depends_on:
                    is_dependency[dep] = True

        worst_idx = -1
        worst_ratio = float('inf')
        for i in range(n):
            if sol[i] == 1 and not is_dependency[i]:
                ratio = features[i].value / (features[i].cost + features[i].time)
                if ratio < worst_ratio:
                    worst_ratio = ratio
                    worst_idx = i

        if worst_idx != -1:
            sol[worst_idx] = 0
        else:
            for i in range(n):
                if sol[i] == 1:
                    ratio = features[i].value / (features[i].cost + features[i].time)
                    if ratio < worst_ratio:
                        worst_ratio = ratio
                        worst_idx = i
            if worst_idx != -1:
                sol[worst_idx] = 0
            else:
                break
    return sol

# ---------- Генетический алгоритм ----------
@dataclass
class GAConfig:
    pop_size: int = 100
    max_gens: int = 200
    tournament_k: int = 3
    p_crossover: float = 0.9
    p_mutation: float = 0.05
    elite_count: int = 2
    method: str = "penalty"
    seed: int = BASE_SEED
    tag: str = "default"

def run_ga(cfg: GAConfig, features: List[Feature], budget: int, deadline: int) -> dict:
    rng = np.random.default_rng(cfg.seed)
    pop = rng.integers(0, 2, size=(cfg.pop_size, N_FEATURES))
    
    def evaluate_population(p):
        if cfg.method == "repair":
            p_repaired = np.array([repair_solution(ind, features, budget, deadline) for ind in p])
            fit = np.array([evaluate_fitness(ind, features, budget, deadline, "repair") for ind in p_repaired])
            return fit, p_repaired
        else:
            fit = np.array([evaluate_fitness(ind, features, budget, deadline, "penalty") for ind in p])
            return fit, p

    fit, pop = evaluate_population(pop)
    
    best_idx = int(np.argmin(fit))
    history_best = [fit[best_idx]]
    elite_archive = [(fit[best_idx], pop[best_idx].copy())]

    for gen in range(1, cfg.max_gens + 1):
        new_pop = []
        order = np.argsort(fit)
        for i in range(cfg.elite_count):
            new_pop.append(pop[order[i]].copy())

        while len(new_pop) < cfg.pop_size:
            idx1 = rng.integers(0, cfg.pop_size, size=cfg.tournament_k)
            i1 = idx1[np.argmin(fit[idx1])]
            idx2 = rng.integers(0, cfg.pop_size, size=cfg.tournament_k)
            i2 = idx2[np.argmin(fit[idx2])]
            
            p1, p2 = pop[i1], pop[i2]
            
            if rng.random() < cfg.p_crossover:
                point = rng.integers(1, N_FEATURES)
                c1 = np.concatenate([p1[:point], p2[point:]])
                c2 = np.concatenate([p2[:point], p1[point:]])
            else:
                c1, c2 = p1.copy(), p2.copy()
                
            for c in [c1, c2]:
                mask = rng.random(N_FEATURES) < cfg.p_mutation
                c[mask] = 1 - c[mask]
                
            new_pop.append(c1)
            if len(new_pop) < cfg.pop_size:
                new_pop.append(c2)
                
        pop = np.array(new_pop)
        fit, pop = evaluate_population(pop)
        
        gen_best_idx = int(np.argmin(fit))
        history_best.append(fit[gen_best_idx])
        if fit[gen_best_idx] < elite_archive[0][0]:
            elite_archive = [(fit[gen_best_idx], pop[gen_best_idx].copy())]

    return {
        "best_f": elite_archive[0][0],
        "best_x": elite_archive[0][1],
        "history_best": history_best
    }

# ---------- Эксперименты и сохранение ----------
def experiment_series(configs: List[GAConfig], features, budget, deadline):
    results = {}
    for cfg in configs:
        print(f"  Запуск конфигурации: {cfg.tag}...")
        runs_best_f = []
        runs_best_x = []
        runs_history = []
        for r in range(N_RUNS):
            c = GAConfig(**{**cfg.__dict__, "seed": cfg.seed + r})
            res = run_ga(c, features, budget, deadline)
            runs_best_f.append(res["best_f"])
            runs_best_x.append(res["best_x"])
            runs_history.append(res["history_best"])
        results[cfg.tag] = {
            "bests_f": np.array(runs_best_f),
            "bests_x": np.array(runs_best_x),
            "histories": np.array(runs_history)
        }
    return results

# НОВАЯ ФУНКЦИЯ ДЛЯ СОХРАНЕНИЯ В CSV
def save_results_csv(results: dict, path: str):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["config", "run", "best_f", "mean_f", "std_f", "median_f", "worst_f"])
        for tag, data in results.items():
            b = data["bests_f"]
            w.writerow([tag, "summary",
                        float(b.min()), float(b.mean()), float(b.std()),
                        float(np.median(b)), float(b.max())])
            for i, v in enumerate(b):
                w.writerow([tag, i, float(v), "", "", "", ""])

def save_histories(results: dict, path: str):
    payload = {}
    for tag, data in results.items():
        payload[f"{tag}__hist"] = data["histories"]
    np.savez_compressed(path, **payload)


def save_best_solution_json(best_x, features, budget, deadline, path: str):
    import json
    selected = [features[i] for i in range(N_FEATURES) if best_x[i] == 1]
    payload = {
        "selected_count": len(selected),
        "budget": budget,
        "deadline": deadline,
        "total_cost": int(sum(f.cost for f in selected)),
        "total_time": int(sum(f.time for f in selected)),
        "total_value": int(sum(f.value for f in selected)),
        "features": [
            {"id": int(f.id), "value": int(f.value), "cost": int(f.cost),
             "time": int(f.time), "depends_on": [int(d) for d in f.depends_on]}
            for f in selected
        ],
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, indent=2)


def plot_flowchart(path: str):
    """Блок-схема бинарного генетического алгоритма с обработкой ограничений."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12.5, 11.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 11.8)
    ax.axis("off")

    def box(x, y, w, h, text, color="#EAF2FB", edge="#2F5D8A", fs=10):
        p = FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.08,rounding_size=0.15",
                           linewidth=1.6, edgecolor=edge, facecolor=color)
        ax.add_patch(p)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
                fontsize=fs, color="#12263A")

    def arrow(x1, y1, x2, y2, text="", color="#2F5D8A", fs=8.5, ha="left"):
        a = FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=15,
                            linewidth=1.6, color=color,
                            connectionstyle="arc3,rad=0.0")
        ax.add_patch(a)
        if text:
            ax.text((x1 + x2) / 2 + 0.15, (y1 + y2) / 2, text, fontsize=fs,
                    color=color, ha=ha, va="center")

    ax.text(7, 11.4, "Блок-схема ЛР №2: ГА для выбора функций ПО (бинарное кодирование)",
            ha="center", va="center", fontsize=14.5, fontweight="bold", color="#12263A")

    cx, w = 4.0, 5.7
    steps = [
        ("Старт\n(seed = 42…61)", 10.5, "#EAF2FB", "#2F5D8A"),
        ("Инициализация бинарной популяции\n(40 генов, 100 особей)", 9.6, "#EAF2FB", "#2F5D8A"),
        ("Оценка фитнеса:\nсуммарная ценность − штрафы", 8.7, "#EAF2FB", "#2F5D8A"),
        ("Обработка ограничений:\nремонт или штрафы", 7.5, "#FEF9E7", "#B7950B"),
        ("Элитизм: 2 лучшие особи", 6.6, "#EAF2FB", "#2F5D8A"),
        ("Турнирная селекция родителей (k = 3)", 5.7, "#EAF2FB", "#2F5D8A"),
        ("Одноточечный кроссовер (p = 0.9)", 4.8, "#EAF2FB", "#2F5D8A"),
        ("Битовая мутация (p = 0.05)", 3.9, "#EAF2FB", "#2F5D8A"),
        ("Ремонт / штраф → допустимость\n(бюджет, срок, зависимости)", 2.9, "#FEF9E7", "#B7950B"),
        ("Новая популяция и оценка фитнеса", 2.0, "#EAF2FB", "#2F5D8A"),
        ("Достигнуто 200 поколений?", 1.1, "#EAF2FB", "#2F5D8A"),
        ("Лучший набор функций\nи отчёт", 0.1, "#E8F8F5", "#117A65"),
    ]
    for text, y, color, edge in steps:
        box(cx, y, w, 0.72, text, color=color, edge=edge)

    for i in range(len(steps) - 2):
        y_from = steps[i][1]
        y_to = steps[i + 1][1] + 0.72
        arrow(cx + w / 2, y_from, cx + w / 2, y_to)

    arrow(cx + w / 2, 1.1, cx + w / 2, 0.82, text="да", ha="left")

    loop_x = 12.0
    arrow(cx + w, 1.46, loop_x, 1.46, color="#B03A2E")
    arrow(loop_x, 1.46, loop_x, 6.96, color="#B03A2E")
    arrow(loop_x, 6.96, cx + w, 6.96, color="#B03A2E")
    ax.text(loop_x + 0.15, 4.3, "нет:\nследующее\nпоколение", fontsize=9,
            color="#B03A2E", ha="left", va="center")

    plt.tight_layout()
    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def plot_dashboard(results: dict, budget: int, deadline: int, features: List[Feature], path: str):
    """Дашборд ЛР №2: сходимость, разброс, средние и состав лучшего решения."""
    from matplotlib.gridspec import GridSpec

    tags = list(results.keys())
    colors = ["#C44E52", "#55A868"]
    fig = plt.figure(figsize=(17, 10.5))
    gs = GridSpec(2, 3, figure=fig, hspace=0.40, wspace=0.30,
                  height_ratios=[1.05, 1.0])
    fig.suptitle("ЛР №2 · ГА для дискретной задачи с ограничениями · 20 независимых запусков",
                 fontsize=16, fontweight="bold", color="#12263A", y=0.98)

    ax1 = fig.add_subplot(gs[0, :2])
    for tag, color in zip(tags, colors):
        hb = results[tag]["histories"]
        mean_curve = -hb.mean(axis=0)
        std_curve = hb.std(axis=0)
        gens = np.arange(len(mean_curve))
        ax1.plot(gens, mean_curve, color=color, label=tag, linewidth=2)
        ax1.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve,
                         color=color, alpha=0.13)
    ax1.set_xlabel("Поколение")
    ax1.set_ylabel("Ценность решения (−fitness)")
    ax1.set_title("Сходимость: среднее ± std по 20 запускам (выше — лучше)")
    ax1.grid(alpha=0.3)
    ax1.legend(fontsize=9)

    ax2 = fig.add_subplot(gs[0, 2])
    data = [-results[t]["bests_f"] for t in tags]
    bp = ax2.boxplot(data, patch_artist=True, widths=0.55)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.72)
    ax2.set_xticklabels(tags, rotation=20, ha="right", fontsize=8.5)
    ax2.set_ylabel("Значение целевой функции")
    ax2.set_title("Разброс по запускам")
    ax2.grid(alpha=0.3, axis="y")

    ax3 = fig.add_subplot(gs[1, :2])
    means = [-results[t]["bests_f"].mean() for t in tags]
    stds = [results[t]["bests_f"].std() for t in tags]
    xpos = np.arange(len(means))
    bars = ax3.bar(xpos, means, yerr=stds, capsize=6, color=colors, alpha=0.82,
                   edgecolor="white", width=0.45)
    for bar, m in zip(bars, means):
        ax3.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 8,
                 f"{m:.1f}", ha="center", va="bottom", fontsize=10, color="#12263A")
    ax3.set_xticks(xpos)
    ax3.set_xticklabels(tags)
    ax3.set_ylabel("Средняя ценность (−fitness)")
    ax3.set_title("Средний результат методов обработки ограничений")
    ax3.set_ylim(min(means) * 0.9, max(means) * 1.05)
    ax3.grid(alpha=0.3, axis="y")

    ax4 = fig.add_subplot(gs[1, 2])
    ax4.axis("off")
    best_repair_idx = int(np.argmin(results["Repair (Ремонт)"]["bests_f"]))
    best_x = results["Repair (Ремонт)"]["bests_x"][best_repair_idx]
    selected = [i for i in range(N_FEATURES) if best_x[i] == 1]
    total_cost = sum(features[i].cost for i in selected)
    total_time = sum(features[i].time for i in selected)
    total_value = sum(features[i].value for i in selected)
    lines = [
        "Лучшее решение (Repair)",
        "",
        f"Функций: {len(selected)} из {N_FEATURES}",
        f"Ценность: {total_value}",
        f"Стоимость: {total_cost} / {budget}",
        f"Время: {total_time} / {deadline}",
        "",
        "Выбранные функции:",
        ", ".join(str(i) for i in selected),
    ]
    ax4.text(0.02, 0.98, "\n".join(lines), ha="left", va="top", fontsize=10,
             family="DejaVu Sans", color="#12263A",
             bbox=dict(boxstyle="round,pad=0.6", facecolor="#E8F8F5", edgecolor="#117A65"))
    ax4.set_title("Интерпретация результата", pad=12)

    plt.savefig(path, dpi=160, bbox_inches="tight")
    plt.close()


def print_best_solution(best_x: np.ndarray, features: List[Feature], budget: int, deadline: int):
    print("\n=== Лучшее найденное решение (Понятное пользователю) ===")
    selected = [features[i] for i in range(N_FEATURES) if best_x[i] == 1]
    total_cost = sum(f.cost for f in selected)
    total_time = sum(f.time for f in selected)
    total_value = sum(f.value for f in selected)
    
    print(f"Выбрано функций: {len(selected)} из {N_FEATURES}")
    print(f"Суммарная стоимость: {total_cost} (Бюджет: {budget})")
    print(f"Суммарное время: {total_time} (Дедлайн: {deadline})")
    print(f"Суммарная ценность: {total_value}")
    print("\nСписок выбранных функций:")
    for f in selected:
        dep_str = f", зависит от: {f.depends_on}" if f.depends_on else ""
        print(f"  - Функция {f.id}: ценность={f.value}, стоимость={f.cost}, время={f.time}{dep_str}")

def plot_convergence(results, path):
    plt.figure(figsize=(10, 5))
    for tag, data in results.items():
        hb = data["histories"]
        mean_curve = hb.mean(axis=0)
        std_curve = hb.std(axis=0)
        gens = np.arange(len(mean_curve))
        plt.plot(gens, mean_curve, label=f"{tag} (mean fitness)")
        plt.fill_between(gens, mean_curve - std_curve, mean_curve + std_curve, alpha=0.15)
    plt.xlabel("Поколение")
    plt.ylabel("Фитнес (чем меньше, тем лучше)")
    plt.title("Сходимость ГА (Сравнение методов работы с ограничениями)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def main():
    os.makedirs("lab2_out", exist_ok=True)
    
    features, budget, deadline = generate_data(BASE_SEED)
    print(f"Сгенерировано {N_FEATURES} функций. Бюджет: {budget}, Дедлайн: {deadline}")
    
    cfgs = [
        GAConfig(method="penalty", tag="Penalty (Штрафы)"),
        GAConfig(method="repair", tag="Repair (Ремонт)")
    ]
    
    print("Запуск серии экспериментов (20 запусков)...")
    results = experiment_series(cfgs, features, budget, deadline)
    
    # Сохраняем CSV
    save_results_csv(results, "lab2_out/lab2_results.csv")
    save_histories(results, "lab2_out/lab2_histories.npz")
    plot_flowchart("lab2_out/lab2_block_schema.png")
    
    print("\n=== Сводка по 20 запускам (Фитнес) ===")
    for tag, data in results.items():
        b = data["bests_f"]
        print(f"{tag:<20} | min: {b.min():.1f} | mean: {b.mean():.1f} | max: {b.max():.1f}")
        
    best_repair_idx = np.argmin(results["Repair (Ремонт)"]["bests_f"])
    best_repair_x = results["Repair (Ремонт)"]["bests_x"][best_repair_idx]
    print_best_solution(best_repair_x, features, budget, deadline)
    save_best_solution_json(best_repair_x, features, budget, deadline,
                            "lab2_out/lab2_best_solution.json")
    
    plot_convergence(results, "lab2_out/lab2_convergence.png")
    plot_dashboard(results, budget, deadline, features, "lab2_out/lab2_dashboard.png")
    print("\nФайлы сохранены в папку lab2_out/ (CSV, JSON, PNG, NPZ)")

if __name__ == "__main__":
    main()