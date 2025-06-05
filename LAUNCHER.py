import pandas as pd
import matplotlib.pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

from ALL import *

def batch_analysis(func, test_data_func, batch_sizes, attempts=5):
    results = []

    print("Анализ производительности SGD с разными размерами батчей")
    print("=" * 60)

    for batch_size in batch_sizes:
        print(f"\nТестирование batch_size = {batch_size}")

        batch_results = {
            'batch_size': batch_size,
            'r2_scores': [],
            'iterations': [],
            'performance_metrics': []
        }

        for attempt in range(attempts):
            x, k, perf_report = sgd_with_performance(
                func,
                armijo_rule_cond(0.3, 0.5, 1e-3),
                batch_size=batch_size,
                limit=1000
            )

            targets_predict = []
            for row in test_data_func.features:
                y_pr = x[0] + row @ x[1:]
                targets_predict.append(y_pr)

            targets = test_data_func.targets.tolist()
            r2_score = metrics.r2_score(targets, targets_predict)

            batch_results['r2_scores'].append(r2_score)
            batch_results['iterations'].append(k)
            batch_results['performance_metrics'].append(perf_report)

        avg_r2 = np.mean(batch_results['r2_scores'])
        avg_iterations = np.mean(batch_results['iterations'])
        avg_operations = np.mean([p['total_ops'] for p in batch_results['performance_metrics']])
        avg_memory = np.mean([p['memory_mb'] for p in batch_results['performance_metrics']])
        avg_time = np.mean([p['time_sec'] for p in batch_results['performance_metrics']])

        result_summary = {
            'batch_size': batch_size,
            'avg_r2_score': avg_r2,
            'avg_iterations': avg_iterations,
            'avg_total_operations': avg_operations,
            'avg_memory_mb': avg_memory,
            'avg_time_sec': avg_time,
            'ops_per_iteration': avg_operations / avg_iterations if avg_iterations > 0 else 0,
            'detailed_results': batch_results
        }

        results.append(result_summary)

        print(f"  Средний R² score: {avg_r2:.4f}")
        print(f"  Среднее количество итераций: {avg_iterations:.1f}")
        print(f"  Среднее количество операций: {avg_operations:.0f}")
        print(f"  Операций на итерацию: {result_summary['ops_per_iteration']:.0f}")
        print(f"  Среднее использование памяти: {avg_memory:.1f} MB")
        print(f"  Среднее время выполнения: {avg_time:.3f} сек")

    return results

def hyperparameter_analysis(data_train, data_test, hyperparams, attempts=3):
    results = []
    print("Анализ производительности SGD с разными гиперпараметрами")
    print("=" * 70)

    for config in hyperparams:
        name = config['name']
        loss_func = config['loss_func']
        learning_rate = config['learning_rate']

        print(f"\nТестирование конфигурации: {name}")

        func = LossFunc(data_train, loss_func)
        test_func = LossFunc(data_test, loss_func)
        config_results = {
            'name': name,
            'r2_scores': [],
            'iterations': [],
            'performance_metrics': []
        }

        for attempt in range(attempts):
            x, k, perf_report = sgd_with_performance(
                func,
                constant(learning_rate),
                batch_size=20,
                limit=1000
            )
            targets_predict = []
            for row in test_func.features:
                y_pr = x[0] + row @ x[1:]
                targets_predict.append(y_pr)

            targets = test_func.targets.tolist()
            r2_score = metrics.r2_score(targets, targets_predict)

            config_results['r2_scores'].append(r2_score)
            config_results['iterations'].append(k)
            config_results['performance_metrics'].append(perf_report)

        avg_r2 = np.mean(config_results['r2_scores'])
        avg_iterations = np.mean(config_results['iterations'])
        avg_operations = np.mean([p['total_ops'] for p in config_results['performance_metrics']])
        avg_memory = np.mean([p['memory_mb'] for p in config_results['performance_metrics']])
        avg_time = np.mean([p['time_sec'] for p in config_results['performance_metrics']])

        result_summary = {
            'name': name,
            'config': config,
            'avg_r2_score': avg_r2,
            'avg_iterations': avg_iterations,
            'avg_total_operations': avg_operations,
            'avg_memory_mb': avg_memory,
            'avg_time_sec': avg_time,
            'ops_per_iteration': avg_operations / avg_iterations if avg_iterations > 0 else 0,
            'detailed_results': config_results
        }

        results.append(result_summary)

        print(f"  Средний R² score: {avg_r2:.4f}")
        print(f"  Среднее количество итераций: {avg_iterations:.1f}")
        print(f"  Среднее количество операций: {avg_operations:.0f}")
        print(f"  Операций на итерацию: {result_summary['ops_per_iteration']:.0f}")
        print(f"  Среднее использование памяти: {avg_memory:.1f} MB")
        print(f"  Среднее время выполнения: {avg_time:.3f} сек")

    return results

def learning_rate_analysis(data_train, data_test, learning_rates, attempts=3):
    results = []
    print("Анализ производительности SGD с разными learning rate")
    print("=" * 60)

    for lr in learning_rates:
        print(f"\nТестирование learning_rate = {lr}")
        func = LossFunc(data_train, mse)
        test_func = LossFunc(data_test, mse)

        lr_results = {
            'learning_rate': lr,
            'r2_scores': [],
            'iterations': [],
            'performance_metrics': []
        }

        for attempt in range(attempts):
            x, k, perf_report = sgd_with_performance(
                func,
                constant(lr),
                batch_size=20,
                limit=1000
            )
            targets_predict = []
            for row in test_func.features:
                y_pr = x[0] + row @ x[1:]
                targets_predict.append(y_pr)
            targets = test_func.targets.tolist()
            r2_score = metrics.r2_score(targets, targets_predict)
            lr_results['r2_scores'].append(r2_score)
            lr_results['iterations'].append(k)
            lr_results['performance_metrics'].append(perf_report)

        avg_r2 = np.mean(lr_results['r2_scores'])
        avg_iterations = np.mean(lr_results['iterations'])
        avg_operations = np.mean([p['total_ops'] for p in lr_results['performance_metrics']])
        avg_memory = np.mean([p['memory_mb'] for p in lr_results['performance_metrics']])
        avg_time = np.mean([p['time_sec'] for p in lr_results['performance_metrics']])

        result_summary = {
            'learning_rate': lr,
            'avg_r2_score': avg_r2,
            'avg_iterations': avg_iterations,
            'avg_total_operations': avg_operations,
            'avg_memory_mb': avg_memory,
            'avg_time_sec': avg_time,
            'ops_per_iteration': avg_operations / avg_iterations if avg_iterations > 0 else 0,
            'detailed_results': lr_results
        }

        results.append(result_summary)

        print(f"  Средний R² score: {avg_r2:.4f}")
        print(f"  Среднее количество итераций: {avg_iterations:.1f}")
        print(f"  Среднее количество операций: {avg_operations:.0f}")
        print(f"  Операций на итерацию: {result_summary['ops_per_iteration']:.0f}")
        print(f"  Среднее использование памяти: {avg_memory:.1f} MB")
        print(f"  Среднее время выполнения: {avg_time:.3f} сек")

    return results

def momentum_analysis(data_train, data_test, momentum_values, attempts=3):
    results = []
    print("Анализ производительности SGD с моментумом")
    print("=" * 60)

    for momentum in momentum_values:
        print(f"\nТестирование momentum = {momentum}")
        func = LossFunc(data_train, mse)
        test_func = LossFunc(data_test, mse)

        momentum_results = {
            'momentum': momentum,
            'r2_scores': [],
            'iterations': [],
            'performance_metrics': []
        }

        for attempt in range(attempts):
            x, k, perf_report = sgd_momentum_with_performance(
                func,
                constant(0.00013),
                momentum=momentum,
                batch_size=20,
                limit=1000
            )

            targets_predict = []
            for row in test_func.features:
                y_pr = x[0] + row @ x[1:]
                targets_predict.append(y_pr)

            targets = test_func.targets.tolist()
            r2_score = metrics.r2_score(targets, targets_predict)

            momentum_results['r2_scores'].append(r2_score)
            momentum_results['iterations'].append(k)
            momentum_results['performance_metrics'].append(perf_report)

        avg_r2 = np.mean(momentum_results['r2_scores'])
        avg_iterations = np.mean(momentum_results['iterations'])
        avg_operations = np.mean([p['total_ops'] for p in momentum_results['performance_metrics']])
        avg_memory = np.mean([p['memory_mb'] for p in momentum_results['performance_metrics']])
        avg_time = np.mean([p['time_sec'] for p in momentum_results['performance_metrics']])

        result_summary = {
            'momentum': momentum,
            'avg_r2_score': avg_r2,
            'avg_iterations': avg_iterations,
            'avg_total_operations': avg_operations,
            'avg_memory_mb': avg_memory,
            'avg_time_sec': avg_time,
            'ops_per_iteration': avg_operations / avg_iterations if avg_iterations > 0 else 0,
            'detailed_results': momentum_results
        }

        results.append(result_summary)

        print(f"  Средний R² score: {avg_r2:.4f}")
        print(f"  Среднее количество итераций: {avg_iterations:.1f}")
        print(f"  Среднее количество операций: {avg_operations:.0f}")
        print(f"  Операций на итерацию: {result_summary['ops_per_iteration']:.0f}")
        print(f"  Среднее использование памяти: {avg_memory:.1f} MB")
        print(f"  Среднее время выполнения: {avg_time:.3f} сек")

    return results

def plot_performance_analysis(results):
    batch_sizes = [r['batch_size'] for r in results]
    r2_scores = [r['avg_r2_score'] for r in results]
    total_ops = [r['avg_total_operations'] for r in results]
    memory_usage = [r['avg_memory_mb'] for r in results]
    execution_times = [r['avg_time_sec'] for r in results]
    ops_per_iter = [r['ops_per_iteration'] for r in results]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.plot(batch_sizes, r2_scores, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Размер батча')
    ax1.set_ylabel('R² score')
    ax1.set_title('Качество модели vs Размер батча')
    ax1.grid(True, alpha=0.3)

    ax2.plot(batch_sizes, total_ops, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Размер батча')
    ax2.set_ylabel('Общее количество операций')
    ax2.set_title('Вычислительная сложность vs Размер батча')
    ax2.grid(True, alpha=0.3)

    ax3.plot(batch_sizes, memory_usage, 'go-', linewidth=2, markersize=6)
    ax3.set_xlabel('Размер батча')
    ax3.set_ylabel('Использование памяти (MB)')
    ax3.set_title('Потребление памяти vs Размер батча')
    ax3.grid(True, alpha=0.3)

    ax4.plot(batch_sizes, execution_times, 'mo-', linewidth=2, markersize=6)
    ax4.set_xlabel('Размер батча')
    ax4.set_ylabel('Время выполнения (сек)')
    ax4.set_title('Время выполнения vs Размер батча')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    plt.plot(batch_sizes, ops_per_iter, 'co-', linewidth=2, markersize=6)
    plt.xlabel('Размер батча')
    plt.ylabel('Операций на итерацию')
    plt.title('Вычислительная сложность на итерацию vs Размер батча')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_hyperparameter_analysis(results):
    names = [r['name'] for r in results]
    r2_scores = [r['avg_r2_score'] for r in results]
    total_ops = [r['avg_total_operations'] for r in results]
    memory_usage = [r['avg_memory_mb'] for r in results]
    execution_times = [r['avg_time_sec'] for r in results]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))

    x_pos = np.arange(len(names))

    ax1.bar(x_pos, r2_scores, color=['blue', 'red', 'green', 'orange', 'purple'][:len(names)])
    ax1.set_xlabel('Конфигурация')
    ax1.set_ylabel('R² score')
    ax1.set_title('Качество модели по конфигурациям')
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(names, rotation=45, ha='right')
    ax1.grid(True, alpha=0.3, axis='y')

    ax2.bar(x_pos, total_ops, color=['blue', 'red', 'green', 'orange', 'purple'][:len(names)])
    ax2.set_xlabel('Конфигурация')
    ax2.set_ylabel('Общее количество операций')
    ax2.set_title('Вычислительная сложность по конфигурациям')
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(names, rotation=45, ha='right')
    ax2.grid(True, alpha=0.3, axis='y')

    ax3.bar(x_pos, memory_usage, color=['blue', 'red', 'green', 'orange', 'purple'][:len(names)])
    ax3.set_xlabel('Конфигурация')
    ax3.set_ylabel('Использование памяти (MB)')
    ax3.set_title('Потребление памяти по конфигурациям')
    ax3.set_xticks(x_pos)
    ax3.set_xticklabels(names, rotation=45, ha='right')
    ax3.grid(True, alpha=0.3, axis='y')

    ax4.bar(x_pos, execution_times, color=['blue', 'red', 'green', 'orange', 'purple'][:len(names)])
    ax4.set_xlabel('Конфигурация')
    ax4.set_ylabel('Время выполнения (сек)')
    ax4.set_title('Время выполнения по конфигурациям')
    ax4.set_xticks(x_pos)
    ax4.set_xticklabels(names, rotation=45, ha='right')
    ax4.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()

def plot_learning_rate_analysis(results):
    learning_rates = [r['learning_rate'] for r in results]
    r2_scores = [r['avg_r2_score'] for r in results]
    total_ops = [r['avg_total_operations'] for r in results]
    memory_usage = [r['avg_memory_mb'] for r in results]
    execution_times = [r['avg_time_sec'] for r in results]

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 10))

    ax1.semilogx(learning_rates, r2_scores, 'bo-', linewidth=2, markersize=6)
    ax1.set_xlabel('Learning Rate')
    ax1.set_ylabel('R² score')
    ax1.set_title('Качество модели vs Learning Rate')
    ax1.grid(True, alpha=0.3)

    ax2.semilogx(learning_rates, total_ops, 'ro-', linewidth=2, markersize=6)
    ax2.set_xlabel('Learning Rate')
    ax2.set_ylabel('Общее количество операций')
    ax2.set_title('Вычислительная сложность vs Learning Rate')
    ax2.grid(True, alpha=0.3)

    ax3.semilogx(learning_rates, memory_usage, 'go-', linewidth=2, markersize=6)
    ax3.set_xlabel('Learning Rate')
    ax3.set_ylabel('Использование памяти (MB)')
    ax3.set_title('Потребление памяти vs Learning Rate')
    ax3.grid(True, alpha=0.3)

    ax4.semilogx(learning_rates, execution_times, 'mo-', linewidth=2, markersize=6)
    ax4.set_xlabel('Learning Rate')
    ax4.set_ylabel('Время выполнения (сек)')
    ax4.set_title('Время выполнения vs Learning Rate')
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

def print_detailed_comparison(results):
    print("\n" + "=" * 80)
    print("ДЕТАЛЬНОЕ СРАВНЕНИЕ ПРОИЗВОДИТЕЛЬНОСТИ")
    print("=" * 80)

    print(
        f"{'Batch Size':<12} "
        f"{'R² Score':<10} "
        f"{'Итерации':<10} "
        f"{'Всего Ops':<12} "
        f"{'Ops/Iter':<10} "
        f"{'Memory MB':<12} "
        f"{'Time (s)':<10}"
    )
    print("-" * 80)

    for result in results:
        print(f"{result['batch_size']:<12} "
              f"{result['avg_r2_score']:<10.4f} "
              f"{result['avg_iterations']:<10.1f} "
              f"{result['avg_total_operations']:<12.0f} "
              f"{result['ops_per_iteration']:<10.0f} "
              f"{result['avg_memory_mb']:<12.1f} "
              f"{result['avg_time_sec']:<10.3f}")

    best_r2 = max(results, key=lambda x: x['avg_r2_score'])
    fastest = min(results, key=lambda x: x['avg_time_sec'])
    most_efficient = min(results, key=lambda x: x['avg_total_operations'])
    least_memory = min(results, key=lambda x: x['avg_memory_mb'])

    print("\n" + "=" * 50)
    print("ОПТИМАЛЬНЫЕ ПАРАМЕТРЫ:")
    print("=" * 50)
    print(f"Лучший R² score: batch_size = {best_r2['batch_size']} (R² = {best_r2['avg_r2_score']:.4f})")
    print(f"Самый быстрый: batch_size = {fastest['batch_size']} ({fastest['avg_time_sec']:.3f} сек)")
    print(
        f"Наименее операций: batch_size = {most_efficient['batch_size']} ({most_efficient['avg_total_operations']:.0f} ops)")
    print(f"Наименее памяти: batch_size = {least_memory['batch_size']} ({least_memory['avg_memory_mb']:.1f} MB)")

def print_detailed_hyperparameter_comparison(results):
    print("\n" + "=" * 90)
    print("ДЕТАЛЬНОЕ СРАВНЕНИЕ ГИПЕРПАРАМЕТРОВ")
    print("=" * 90)

    print(
        f"{'Конфигурация':<20} "
        f"{'R² Score':<10} "
        f"{'Итерации':<10} "
        f"{'Всего Ops':<12} "
        f"{'Ops/Iter':<10} "
        f"{'Memory MB':<12} "
        f"{'Time (s)':<10}"
    )
    print("-" * 90)

    for result in results:
        print(f"{result['name']:<20} "
              f"{result['avg_r2_score']:<10.4f} "
              f"{result['avg_iterations']:<10.1f} "
              f"{result['avg_total_operations']:<12.0f} "
              f"{result['ops_per_iteration']:<10.0f} "
              f"{result['avg_memory_mb']:<12.1f} "
              f"{result['avg_time_sec']:<10.3f}")

    best_r2 = max(results, key=lambda x: x['avg_r2_score'])
    fastest = min(results, key=lambda x: x['avg_time_sec'])
    most_efficient = min(results, key=lambda x: x['avg_total_operations'])
    least_memory = min(results, key=lambda x: x['avg_memory_mb'])

    print("\n" + "=" * 60)
    print("ОПТИМАЛЬНЫЕ КОНФИГУРАЦИИ:")
    print("=" * 60)
    print(f"Лучший R² score: {best_r2['name']} (R² = {best_r2['avg_r2_score']:.4f})")
    print(f"Самый быстрый: {fastest['name']} ({fastest['avg_time_sec']:.3f} сек)")
    print(f"Наименее операций: {most_efficient['name']} ({most_efficient['avg_total_operations']:.0f} ops)")
    print(f"Наименее памяти: {least_memory['name']} ({least_memory['avg_memory_mb']:.1f} MB)")

def print_detailed_learning_rate_comparison(results):
    print("\n" + "=" * 90)
    print("ДЕТАЛЬНОЕ СРАВНЕНИЕ LEARNING RATES")
    print("=" * 90)

    print(
        f"{'Learning Rate':<15} "
        f"{'R² Score':<10} "
        f"{'Итерации':<10} "
        f"{'Всего Ops':<12} "
        f"{'Ops/Iter':<10} "
        f"{'Memory MB':<12} "
        f"{'Time (s)':<10}"
    )
    print("-" * 90)

    for result in results:
        print(f"{result['learning_rate']:<15} "
              f"{result['avg_r2_score']:<10.4f} "
              f"{result['avg_iterations']:<10.1f} "
              f"{result['avg_total_operations']:<12.0f} "
              f"{result['ops_per_iteration']:<10.0f} "
              f"{result['avg_memory_mb']:<12.1f} "
              f"{result['avg_time_sec']:<10.3f}")

    best_r2 = max(results, key=lambda x: x['avg_r2_score'])
    fastest = min(results, key=lambda x: x['avg_time_sec'])
    most_efficient = min(results, key=lambda x: x['avg_total_operations'])
    least_memory = min(results, key=lambda x: x['avg_memory_mb'])

    print("\n" + "=" * 60)
    print("ОПТИМАЛЬНЫЕ LEARNING RATES:")
    print("=" * 60)
    print(f"Лучший R² score: {best_r2['learning_rate']} (R² = {best_r2['avg_r2_score']:.4f})")
    print(f"Самый быстрый: {fastest['learning_rate']} ({fastest['avg_time_sec']:.3f} сек)")
    print(f"Наименее операций: {most_efficient['learning_rate']} ({most_efficient['avg_total_operations']:.0f} ops)")
    print(f"Наименее памяти: {least_memory['learning_rate']} ({least_memory['avg_memory_mb']:.1f} MB)")

if __name__ == "__main__":
    try:
        df = pd.read_csv("data/Student_Performance.csv", header=None)
    except FileNotFoundError:
        print("Файл data/Student_Performance.csv не найден!")
        print("Создаем синтетические данные для демонстрации")

        np.random.seed(42)
        n_samples, n_features = 1000, 5
        X_synthetic = np.random.randn(n_samples, n_features)
        y_synthetic = (3 * X_synthetic[:, 0] + 2 * X_synthetic[:, 1] - X_synthetic[:, 2]
                       + 0.5 * X_synthetic[:, 3] + np.random.randn(n_samples) * 0.1)

        df = pd.DataFrame(np.column_stack([X_synthetic, y_synthetic]))

    data_train, data_test = train_test_split(df, test_size=0.25, random_state=42)

    print("Выберите тип анализа:")
    print("1. Анализ размеров батчей")
    print("2. Анализ гиперпараметров и регуляризации")
    print("3. Анализ learning rates")
    print("4. Анализ momentum (SGD с моментумом)")
    print("5. Все анализы")

    choice = input("Введите номер (1-5): ").strip()

    if choice in ['1', '5']:
        test_data_func = LossFunc(data_test, mse)
        func = LossFunc(data_train, mse)
        batch_sizes_to_test = [1, 5, 10, 20, 50, 100]

        print("Запуск анализа размеров батчей...")
        performance_results = batch_analysis(func, test_data_func, batch_sizes_to_test, attempts=3)
        print_detailed_comparison(performance_results)
        plot_performance_analysis(performance_results)

    if choice in ['2', '5']:
        hyperparams_to_test = [
            {
                'name': 'MSE',
                'loss_func': mse,
                'learning_rate': 0.00013
            },
            {
                'name': 'MSE+L1(0.0013)',
                'loss_func': mse_l1(0.0013),
                'learning_rate': 0.0001
            },
            {
                'name': 'MSE+L1(0.008)',
                'loss_func': mse_l1(0.008),
                'learning_rate': 0.00008
            },
            {
                'name': 'MSE+L2(0.0013)',
                'loss_func': mse_l2(0.0013),
                'learning_rate': 0.00013
            },
            {
                'name': 'MSE+L2(0.008)',
                'loss_func': mse_l2(0.008),
                'learning_rate': 0.00008
            },
            {
                'name': 'ElasticNet(0.001)',
                'loss_func': mse_elastic_net(0.001, 0.001),
                'learning_rate': 0.00013
            }
        ]

        print("Запуск анализа гиперпараметров...")
        hyperparam_results = hyperparameter_analysis(data_train, data_test, hyperparams_to_test, attempts=3)
        print_detailed_hyperparameter_comparison(hyperparam_results)
        plot_hyperparameter_analysis(hyperparam_results)

    if choice in ['3', '5']:
        learning_rates_to_test = [0.00008, 0.0001, 0.00013]

        print("Запуск анализа learning rates...")
        lr_results = learning_rate_analysis(data_train, data_test, learning_rates_to_test, attempts=3)
        print_detailed_learning_rate_comparison(lr_results)
        plot_learning_rate_analysis(lr_results)

    if choice in ['4', '5']:
        momentum_values_to_test = [0.0, 0.5, 0.9, 0.95, 0.99]

        print("Запуск анализа momentum...")
        momentum_results = momentum_analysis(data_train, data_test, momentum_values_to_test, attempts=3)

        print("\n" + "=" * 80)
        print("ДЕТАЛЬНОЕ СРАВНЕНИЕ MOMENTUM VALUES")
        print("=" * 80)

        print(f"{'Momentum':<10} {'R² Score':<10} {'Итерации':<10} {'Всего Ops':<12} {'Time (s)':<10}")
        print("-" * 60)

        for result in momentum_results:
            print(f"{result['momentum']:<10} "
                  f"{result['avg_r2_score']:<10.4f} "
                  f"{result['avg_iterations']:<10.1f} "
                  f"{result['avg_total_operations']:<12.0f} "
                  f"{result['avg_time_sec']:<10.3f}")
