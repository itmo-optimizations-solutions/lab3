import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from numpy import ndarray

from sklearn.preprocessing import StandardScaler
from typing import Tuple, Optional

class RegressionDatasetGenerator:
    def __init__(self, random_state: int = 42):
        self.random_state = random_state
        np.random.seed(random_state)

    @staticmethod
    def generate_linear_regression_data(
        n_samples: int = 1000,
        n_features: int = 5,
        noise_level: float = 0.1,
        true_weights: Optional[np.ndarray] = None,
        correlated_features: bool = False,
        outlier_fraction: float = 0.0
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if correlated_features:
            cov_matrix = np.eye(n_features)
            for i in range(n_features):
                for j in range(i + 1, n_features):
                    cov_matrix[i, j] = cov_matrix[j, i] = 0.3 ** abs(i - j)
            X = np.random.multivariate_normal(np.zeros(n_features), cov_matrix, n_samples)
        else:
            X = np.random.randn(n_samples, n_features)

        if true_weights is None:
            true_weights = np.random.randn(n_features)
            zero_indices = np.random.choice(n_features, size=n_features // 3, replace=False)
            true_weights[zero_indices] = 0

        y = X @ true_weights

        noise = np.random.normal(0, noise_level, n_samples)
        y += noise

        if outlier_fraction > 0:
            n_outliers = int(n_samples * outlier_fraction)
            outlier_indices = np.random.choice(n_samples, n_outliers, replace=False)
            y[outlier_indices] += np.random.normal(0, noise_level * 10, n_outliers)

        return X, y, true_weights

    @staticmethod
    def generate_polynomial_regression_data(
        n_samples: int = 1000,
        degree: int = 3,
        noise_level: float = 0.1,
        x_range: Tuple[float, float] = (-2, 2),
        interaction_terms: bool = False
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        if interaction_terms:
            n_base_features = 2
            X_base = np.random.uniform(x_range[0], x_range[1], (n_samples, n_base_features))
        else:
            n_base_features = 1
            X_base = np.random.uniform(x_range[0], x_range[1], (n_samples, n_base_features))

        from sklearn.preprocessing import PolynomialFeatures
        poly_features = PolynomialFeatures(degree=degree, include_bias=True)
        X = poly_features.fit_transform(X_base)

        n_features = X.shape[1]
        true_c = np.random.randn(n_features) * 0.5

        feature_names = poly_features.get_feature_names_out()
        for i, name in enumerate(feature_names):
            degree_sum = sum(int(char) for char in name if char.isdigit())
            if degree_sum > 1:
                true_c[i] *= 0.3 ** (degree_sum - 1)

        y = X @ true_c

        noise = np.random.normal(0, noise_level * np.std(y), n_samples)
        y += noise

        return X, y, true_c

    @staticmethod
    def generate_nonlinear_regression_data(
        n_samples: int = 1000,
        n_features: int = 3,
        function_type: str = 'sinusoidal',
        noise_level: float = 0.1
    ) -> Tuple[np.ndarray, np.ndarray]:
        X = np.random.uniform(-2, 2, (n_samples, n_features))
        y = 0

        if function_type == 'sinusoidal':
            y = (np.sin(X[:, 0]) +
                 0.5 * np.cos(2 * X[:, 1]) +
                 0.3 * np.sin(X[:, 0] * X[:, 1]))
            if n_features > 2:
                y += 0.2 * np.sin(X[:, 2])

        elif function_type == 'exponential':
            y = (np.exp(-X[:, 0] ** 2) +
                 0.5 * np.exp(-X[:, 1] ** 2))
            if n_features > 2:
                y += 0.3 * np.tanh(X[:, 2])

        elif function_type == 'mixed':
            y = (X[:, 0] ** 2 +
                 np.sin(X[:, 1]) +
                 0.5 * np.log(np.abs(X[:, 0]) + 1))
            if n_features > 2:
                y += 0.3 * X[:, 2] ** 3

        noise = np.random.normal(0, noise_level * np.std(y), n_samples)
        y += noise

        return X, y

    def generate_regression_dataset_with_splits(
        self,
        dataset_type: str = 'linear',
        n_samples: int = 1000,
        test_size: float = 0.2,
        val_size: float = 0.2,
        normalize: bool = True,
        **kwargs
    ) -> dict:
        if dataset_type == 'linear':
            X, y, true_weights = self.generate_linear_regression_data(n_samples, **kwargs)
            extra_info = {'true_weights': true_weights}
        elif dataset_type == 'polynomial':
            true_c: ndarray
            X, y, true_c = self.generate_polynomial_regression_data(n_samples, **kwargs)
            extra_info = {'true_c': true_c}
        elif dataset_type == 'nonlinear':
            X, y = self.generate_nonlinear_regression_data(n_samples, **kwargs)
            extra_info = {}
        else:
            raise ValueError(f"Unknown type: {dataset_type}")

        n_test = int(n_samples * test_size)
        n_val = int((n_samples - n_test) * val_size)
        n_train = n_samples - n_test - n_val

        indices = np.random.permutation(n_samples)

        X_train = X[indices[:n_train]]
        y_train = y[indices[:n_train]]

        X_val = X[indices[n_train:n_train + n_val]]
        y_val = y[indices[n_train:n_train + n_val]]

        X_test = X[indices[n_train + n_val:]]
        y_test = y[indices[n_train + n_val:]]

        if normalize:
            scaler_X = StandardScaler()
            scaler_y = StandardScaler()

            X_train = scaler_X.fit_transform(X_train)
            X_val = scaler_X.transform(X_val)
            X_test = scaler_X.transform(X_test)

            y_train = scaler_y.fit_transform(y_train.reshape(-1, 1)).flatten()
            y_val = scaler_y.transform(y_val.reshape(-1, 1)).flatten()
            y_test = scaler_y.transform(y_test.reshape(-1, 1)).flatten()

            extra_info.update({
                'scaler_X': scaler_X,
                'scaler_y': scaler_y
            })

        return {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'dataset_type': dataset_type,
            'n_features': X.shape[1],
            **extra_info
        }

    @staticmethod
    def visualize_dataset(X: np.ndarray, y: np.ndarray, dataset_type: str = 'linear'):
        if X.shape[1] == 1:
            plt.figure(figsize=(10, 6))
            plt.scatter(X[:, 0], y, alpha=0.6)
            plt.xlabel('Признак X')
            plt.ylabel('Целевая переменная y')
            plt.title(f'Датасет для {dataset_type} регрессии')
            plt.grid(True, alpha=0.3)
            plt.show()

        elif X.shape[1] >= 2:
            fig, axes = plt.subplots(2, 2, figsize=(12, 10))

            axes[0, 0].hist(y, bins=30, alpha=0.7)
            axes[0, 0].set_title('Распределение y')
            axes[0, 0].set_xlabel('y')
            axes[0, 0].set_ylabel('Частота')

            axes[0, 1].scatter(X[:, 0], y, alpha=0.6)
            axes[0, 1].set_title('y vs первый признак')
            axes[0, 1].set_xlabel('Признак 1')
            axes[0, 1].set_ylabel('y')

            axes[1, 0].scatter(X[:, 1], y, alpha=0.6)
            axes[1, 0].set_title('y vs второй признак')
            axes[1, 0].set_xlabel('Признак 2')
            axes[1, 0].set_ylabel('y')

            if X.shape[1] <= 10:  # not all signs
                data_for_corr = np.column_stack([X, y])
                feature_names = [f'X{i}' for i in range(X.shape[1])] + ['y']
                corr_matrix = np.corrcoef(data_for_corr.T)

                im = axes[1, 1].imshow(corr_matrix, cmap='coolwarm', vmin=-1, vmax=1)
                axes[1, 1].set_xticks(range(len(feature_names)))
                axes[1, 1].set_yticks(range(len(feature_names)))
                axes[1, 1].set_xticklabels(feature_names)
                axes[1, 1].set_yticklabels(feature_names)
                axes[1, 1].set_title('Корреляционная матрица')

                for i in range(len(feature_names)):
                    for j in range(len(feature_names)):
                        text = axes[1, 1].text(j, i, f'{corr_matrix[i, j]:.2f}',
                                               ha="center", va="center", color="black")

                plt.colorbar(im, ax=axes[1, 1])
            else:
                axes[1, 1].text(0.5, 0.5, 'Слишком много признаков\nдля отображения корреляций',
                                ha='center', va='center', transform=axes[1, 1].transAxes)

            plt.tight_layout()
            plt.show()

if __name__ == "__main__":
    generator = RegressionDatasetGenerator(random_state=42)

    print("Генерация данных для линейной регрессии")
    linear_data = generator.generate_regression_dataset_with_splits(
        dataset_type='linear',
        n_samples=1000,
        n_features=5,
        noise_level=0.2,
        correlated_features=True,
        outlier_fraction=0.05
    )

    print(f"Размеры train set: X{linear_data['X_train'].shape}, y{linear_data['y_train'].shape}")
    print(f"Размеры val set: X{linear_data['X_val'].shape}, y{linear_data['y_val'].shape}")
    print(f"Размеры test set: X{linear_data['X_test'].shape}, y{linear_data['y_test'].shape}")
    print(f"Истинные веса: {linear_data['true_weights'][:5]}...")

    print("\nГенерация данных для полиномиальной регрессии")
    poly_data = generator.generate_regression_dataset_with_splits(
        dataset_type='polynomial',
        n_samples=1000,
        degree=3,
        noise_level=0.1,
        interaction_terms=True
    )

    print(f"Размеры train set: X{poly_data['X_train'].shape}, y{poly_data['y_train'].shape}")
    print(f"Количество полиномиальных признаков: {poly_data['n_features']}")

    print("\nГенерация данных для нелинейной регрессии")
    nonlinear_data = generator.generate_regression_dataset_with_splits(
        dataset_type='nonlinear',
        n_samples=1000,
        n_features=3,
        function_type='sinusoidal',
        noise_level=0.1
    )

    print(f"Размеры train set: X{nonlinear_data['X_train'].shape}, y{nonlinear_data['y_train'].shape}")

    # MARK: менять вызов функций здесь (для картинок)
    print("\nВизуализация данных")
    X_simple, y_simple, _ = generator.generate_polynomial_regression_data()
    generator.visualize_dataset(X_simple[:, [1]], y_simple, 'nonlinear')  # x^1 for example

    print("\nСоздание DataFrame")
    df_train = pd.DataFrame(linear_data['X_train'],
                            columns=[f'feature_{i}' for i in range(linear_data['X_train'].shape[1])])
    df_train['target'] = linear_data['y_train']
    print("Первые 5 строк train set:")
    print(df_train.head())

    print("\nСтатистика по данным:")
    print(df_train.describe())
