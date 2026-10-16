"""Entrena una CNN mínima y evalúa generalización dejando autores completos fuera.

Preparación:
1. Ejecutar la aplicación y abrir http://127.0.0.1:5001/prepare?size=64.
2. Instalar las dependencias de ML con `python -m pip install -r requirements-ml.txt`.
3. Ejecutar `python train_minimal_cnn.py --leave-one-author-out`.

La clase describe si quien dibujó estaba mirando (`abiertos`) o a ciegas
(`cerrados`); no describe los ojos representados en el dibujo.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.stats import t as student_t

try:
    import torch
    from torch import nn
    from torch.nn import functional as torch_functional
    from torch.utils.data import DataLoader, TensorDataset
except ImportError as error:
    raise SystemExit(
        "Falta PyTorch. Instálalo en este entorno con: "
        "python -m pip install -r requirements-ml.txt"
    ) from error


CLASS_NAMES = ("abiertos", "cerrados")
DEFAULT_SEEDS = (17, 37, 73, 101, 202)


class MinimalCNN(nn.Module):
    """CNN pequeña con GAP antes de la capa de clasificación."""

    def __init__(self, class_count: int = len(CLASS_NAMES)) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(1, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
        )
        self.global_average_pooling = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(64, class_count)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feature_maps = self.features(images)
        pooled = self.global_average_pooling(feature_maps).flatten(start_dim=1)
        return self.classifier(pooled)


def load_prepared_arrays(data_dir: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    paths = {name: data_dir / f"{name}.npy" for name in ("X", "y", "autores")}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "Faltan arreglos preparados: " + ", ".join(missing)
            + ". Inicia la app y visita /prepare?size=64."
        )

    images = np.load(paths["X"], allow_pickle=False)
    labels = np.load(paths["y"], allow_pickle=False).astype(str)
    authors = np.load(paths["autores"], allow_pickle=False).astype(str)

    if images.ndim == 4 and images.shape[-1] == 1:
        images = images[..., 0]
    if images.ndim != 3:
        raise ValueError(f"X.npy debe tener forma (n, alto, ancho); se recibió {images.shape}.")
    if not (len(images) == len(labels) == len(authors)):
        raise ValueError("X.npy, y.npy y autores.npy no tienen la misma cantidad de filas.")
    if set(np.unique(labels)) != set(CLASS_NAMES):
        raise ValueError(
            f"Se necesitan las dos clases {CLASS_NAMES}; encontradas: {sorted(set(labels))}."
        )
    if np.any(~np.isfinite(images)):
        raise ValueError("X.npy contiene valores no finitos.")

    return images, labels, authors


def resolve_test_authors(
    authors: np.ndarray,
    test_author: str | None,
    leave_one_author_out: bool,
) -> list[str]:
    """Elige los autores de test sin seleccionarlos según el rendimiento."""
    available_authors = sorted({str(author) for author in authors})
    if leave_one_author_out:
        if test_author is not None:
            raise ValueError("--test-author y leave-one-author-out no se pueden combinar.")
        if len(available_authors) < 2:
            raise ValueError("Se necesitan al menos dos autores para leave-one-author-out.")
        return available_authors

    selected_author = test_author or "martin"
    if selected_author not in available_authors:
        raise ValueError(f"El autor de prueba '{selected_author}' no existe en los datos.")
    return [selected_author]


def split_by_author(
    labels: np.ndarray,
    authors: np.ndarray,
    test_author: str,
    validation_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Separa al autor de prueba antes de crear train/validation."""
    test_indices = np.flatnonzero(authors == test_author)
    train_pool = np.flatnonzero(authors != test_author)
    if len(test_indices) == 0:
        raise ValueError(f"No hay dibujos del autor de prueba '{test_author}'.")
    if set(np.unique(labels[test_indices])) != set(CLASS_NAMES):
        raise ValueError(
            f"El autor de prueba '{test_author}' debe tener ejemplos de ambas clases."
        )
    if len(train_pool) == 0:
        raise ValueError("No quedan autores para entrenar después de reservar el test.")

    rng = np.random.default_rng(seed)
    train_indices: list[int] = []
    validation_indices: list[int] = []
    for class_name in CLASS_NAMES:
        class_indices = train_pool[labels[train_pool] == class_name]
        if len(class_indices) < 2:
            raise ValueError(
                f"Se necesitan al menos dos ejemplos de '{class_name}' "
                "entre los autores de entrenamiento."
            )
        shuffled = rng.permutation(class_indices)
        validation_count = min(
            len(shuffled) - 1,
            max(1, int(round(len(shuffled) * validation_fraction))),
        )
        validation_indices.extend(shuffled[:validation_count].tolist())
        train_indices.extend(shuffled[validation_count:].tolist())

    return (
        np.asarray(train_indices, dtype=np.int64),
        np.asarray(validation_indices, dtype=np.int64),
        test_indices.astype(np.int64),
    )


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True, warn_only=True)


def make_loader(
    images: np.ndarray,
    labels: np.ndarray,
    indices: np.ndarray,
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> DataLoader:
    image_tensor = torch.from_numpy(
        np.ascontiguousarray(images[indices, None, :, :].astype(np.float32) / 255.0)
    )
    label_lookup = {name: index for index, name in enumerate(CLASS_NAMES)}
    label_tensor = torch.tensor(
        [label_lookup[str(name)] for name in labels[indices]], dtype=torch.long
    )
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(image_tensor, label_tensor),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=0,
        generator=generator,
    )


def validation_loss(model: MinimalCNN, loader: DataLoader, loss_fn: nn.Module) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            total_loss += float(loss_fn(logits, labels).item()) * len(labels)
            total_items += len(labels)
    return total_loss / total_items


def train_one_seed(
    images: np.ndarray,
    labels: np.ndarray,
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    seed: int,
    epochs: int,
    batch_size: int,
    patience: int,
) -> tuple[MinimalCNN, list[dict[str, float]], int]:
    set_seed(seed)
    train_loader = make_loader(images, labels, train_indices, batch_size, True, seed)
    validation_loader = make_loader(
        images, labels, validation_indices, batch_size, False, seed
    )

    model = MinimalCNN()
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    best_state = copy.deepcopy(model.state_dict())
    best_loss = float("inf")
    best_epoch = 0
    epochs_without_improvement = 0
    history: list[dict[str, float]] = []

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss_sum = 0.0
        correct = 0
        total = 0
        for batch_images, batch_labels in train_loader:
            optimizer.zero_grad()
            logits = model(batch_images)
            loss = loss_fn(logits, batch_labels)
            loss.backward()
            optimizer.step()

            train_loss_sum += float(loss.item()) * len(batch_labels)
            correct += int((logits.argmax(dim=1) == batch_labels).sum().item())
            total += len(batch_labels)

        val_loss = validation_loss(model, validation_loader, loss_fn)
        row = {
            "epoch": float(epoch),
            "train_loss": train_loss_sum / total,
            "train_accuracy": correct / total,
            "validation_loss": val_loss,
        }
        history.append(row)
        print(
            f"seed={seed} epoch={epoch:02d} "
            f"loss={row['train_loss']:.4f} acc={row['train_accuracy']:.3f} "
            f"val_loss={val_loss:.4f}"
        )

        if val_loss < best_loss:
            best_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())
            best_epoch = epoch
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                break

    model.load_state_dict(best_state)
    return model, history, best_epoch


def predict(model: MinimalCNN, loader: DataLoader) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probabilities: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    with torch.no_grad():
        for images, _ in loader:
            probs = torch.softmax(model(images), dim=1).cpu().numpy()
            probabilities.append(probs)
            predictions.append(probs.argmax(axis=1))
    return np.concatenate(predictions), np.concatenate(probabilities)


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, object]:
    confusion = np.zeros((len(CLASS_NAMES), len(CLASS_NAMES)), dtype=np.int64)
    for actual, predicted in zip(y_true, y_pred):
        confusion[int(actual), int(predicted)] += 1

    per_class: dict[str, dict[str, float]] = {}
    for class_index, class_name in enumerate(CLASS_NAMES):
        true_positive = int(confusion[class_index, class_index])
        precision_denominator = int(confusion[:, class_index].sum())
        recall_denominator = int(confusion[class_index, :].sum())
        precision = true_positive / precision_denominator if precision_denominator else 0.0
        recall = true_positive / recall_denominator if recall_denominator else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if precision + recall
            else 0.0
        )
        per_class[class_name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

    return {
        "accuracy": float(np.mean(y_true == y_pred)),
        "confusion_matrix": confusion.tolist(),
        "per_class": per_class,
    }


def summarize_runs(runs: list[dict[str, object]]) -> dict[str, object]:
    def interval(values: list[float]) -> dict[str, float | None]:
        mean = float(np.mean(values))
        standard_deviation = float(np.std(values, ddof=1)) if len(values) > 1 else 0.0
        if len(values) < 2:
            lower = upper = None
        else:
            margin = float(student_t.ppf(0.975, df=len(values) - 1))
            margin *= standard_deviation / np.sqrt(len(values))
            # Las métricas son proporciones; intersectamos el IC con su rango válido.
            lower = max(0.0, mean - margin)
            upper = min(1.0, mean + margin)
        return {
            "mean": mean,
            "standard_deviation": standard_deviation,
            "ci95_lower": lower,
            "ci95_upper": upper,
        }

    summary: dict[str, object] = {"accuracy": interval([float(r["accuracy"]) for r in runs])}
    class_summaries: dict[str, dict[str, object]] = {}
    for class_name in CLASS_NAMES:
        class_summaries[class_name] = {
            metric: interval(
                [float(r["per_class"][class_name][metric]) for r in runs]  # type: ignore[index]
            )
            for metric in ("precision", "recall", "f1")
        }
    summary["per_class"] = class_summaries
    return summary


def save_cam_examples(
    model: MinimalCNN,
    images: np.ndarray,
    test_indices: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: Path,
    seed: int,
    examples_per_class: int,
) -> None:
    model.eval()
    for class_index, class_name in enumerate(CLASS_NAMES):
        candidates = np.flatnonzero(y_true == class_index)
        selected = candidates[:examples_per_class]
        for position in selected:
            source_index = int(test_indices[position])
            image_tensor = torch.from_numpy(
                images[source_index][None, None].astype(np.float32) / 255.0
            )
            with torch.no_grad():
                feature_maps = model.features(image_tensor)
                weights = model.classifier.weight[class_index]
                cam = torch.relu((feature_maps[0] * weights[:, None, None]).sum(dim=0))
                cam = torch_functional.interpolate(
                    cam[None, None],
                    size=images.shape[1:],
                    mode="bilinear",
                    align_corners=False,
                )[0, 0].cpu().numpy()

            cam = cam - float(cam.min())
            if float(cam.max()) > 0:
                cam = cam / float(cam.max())
            grayscale = np.clip(images[source_index], 0, 255).astype(np.uint8)
            base = np.repeat(grayscale[:, :, None], 3, axis=2).astype(np.float32)
            heat = np.zeros_like(base)
            heat[:, :, 0] = cam * 255.0
            overlay = np.clip(base * 0.55 + heat * 0.45, 0, 255).astype(np.uint8)
            prediction_name = CLASS_NAMES[int(y_pred[position])]
            filename = (
                f"seed_{seed}_{class_name}_sample_{source_index}_"
                f"pred_{prediction_name}.png"
            )
            Image.fromarray(overlay).save(output_dir / filename)


def write_predictions(
    path: Path,
    indices: np.ndarray,
    authors: np.ndarray,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    probabilities: np.ndarray,
) -> None:
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(
            [
                "sample_index",
                "autor",
                "clase_real",
                "clase_predicha",
                "probabilidad_abiertos",
                "probabilidad_cerrados",
            ]
        )
        for position, source_index in enumerate(indices):
            writer.writerow(
                [
                    int(source_index),
                    authors[source_index],
                    CLASS_NAMES[int(y_true[position])],
                    CLASS_NAMES[int(y_pred[position])],
                    float(probabilities[position, 0]),
                    float(probabilities[position, 1]),
                ]
            )


def run(args: argparse.Namespace) -> None:
    images, labels, authors = load_prepared_arrays(args.data_dir)
    if not 0 < args.validation_fraction < 0.5:
        raise ValueError("--validation-fraction debe estar entre 0 y 0.5.")
    if args.cam_per_class < 0:
        raise ValueError("--cam-per-class no puede ser negativo.")

    test_authors = resolve_test_authors(
        authors,
        test_author=args.test_author,
        leave_one_author_out=args.leave_one_author_out,
    )
    protocol_dir = args.output_dir / (
        "leave_one_author_out"
        if args.leave_one_author_out
        else f"single_author_{test_authors[0]}"
    )
    protocol_dir.mkdir(parents=True, exist_ok=True)
    fold_results: list[dict[str, object]] = []
    out_of_fold: dict[int, dict[str, list[np.ndarray]]] = {
        seed: {"true": [], "predicted": []} for seed in args.seeds
    }

    for test_author in test_authors:
        train_authors = sorted({str(author) for author in authors} - {test_author})
        if not train_authors:
            raise ValueError("No hay autores disponibles para entrenamiento.")

        train_indices, validation_indices, test_indices = split_by_author(
            labels,
            authors,
            test_author,
            args.validation_fraction,
            args.split_seed,
        )
        test_labels = np.asarray(
            [CLASS_NAMES.index(str(label)) for label in labels[test_indices]],
            dtype=np.int64,
        )
        fold_dir = protocol_dir / f"test_{test_author}"
        fold_runs: list[dict[str, object]] = []

        for seed in args.seeds:
            model, history, best_epoch = train_one_seed(
                images,
                labels,
                train_indices,
                validation_indices,
                seed,
                args.epochs,
                args.batch_size,
                args.patience,
            )
            test_loader = make_loader(
                images, labels, test_indices, args.batch_size, False, seed
            )
            predictions, probabilities = predict(model, test_loader)
            metrics = calculate_metrics(test_labels, predictions)
            run_dir = fold_dir / f"seed_{seed}"
            run_dir.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                "model_state_dict": model.state_dict(),
                "class_names": CLASS_NAMES,
                "test_author": test_author,
                "train_authors": train_authors,
                "seed": seed,
                "split_seed": args.split_seed,
                "input_shape": list(images.shape[1:]),
                "architecture": "3 conv blocks -> GlobalAveragePooling -> 2-class linear head",
                "best_epoch": best_epoch,
            }
            torch.save(checkpoint, run_dir / "model.pt")
            with (run_dir / "history.json").open("w", encoding="utf-8") as file:
                json.dump(history, file, ensure_ascii=False, indent=2)
            write_predictions(
                run_dir / "predictions.csv",
                test_indices,
                authors,
                test_labels,
                predictions,
                probabilities,
            )
            if args.cam_per_class:
                cam_dir = run_dir / "cam"
                cam_dir.mkdir(parents=True, exist_ok=True)
                save_cam_examples(
                    model,
                    images,
                    test_indices,
                    test_labels,
                    predictions,
                    cam_dir,
                    seed,
                    args.cam_per_class,
                )

            fold_runs.append({"seed": seed, **metrics, "best_epoch": best_epoch})
            out_of_fold[seed]["true"].append(test_labels)
            out_of_fold[seed]["predicted"].append(predictions)
            print(
                f"test_author={test_author} seed={seed} "
                f"test_accuracy={float(metrics['accuracy']):.3f}"
            )

        train_class_counts = {
            class_name: int(np.sum(labels[train_indices] == class_name))
            for class_name in CLASS_NAMES
        }
        test_class_counts = {
            class_name: int(np.sum(labels[test_indices] == class_name))
            for class_name in CLASS_NAMES
        }
        fold_results.append(
            {
                "test_author": test_author,
                "train_authors": train_authors,
                "train_examples": int(len(train_indices)),
                "validation_examples": int(len(validation_indices)),
                "test_examples": int(len(test_indices)),
                "train_class_counts": train_class_counts,
                "test_class_counts": test_class_counts,
                "split_seed": args.split_seed,
                "runs": fold_runs,
                "summary_across_seeds": summarize_runs(fold_runs),
            }
        )

    out_of_fold_runs: list[dict[str, object]] = []
    for seed in args.seeds:
        y_true = np.concatenate(out_of_fold[seed]["true"])
        y_pred = np.concatenate(out_of_fold[seed]["predicted"])
        out_of_fold_runs.append(
            {"seed": seed, **calculate_metrics(y_true, y_pred)}
        )

    fold_mean_runs: list[dict[str, object]] = []
    for fold in fold_results:
        fold_summary = fold["summary_across_seeds"]
        fold_mean_runs.append(
            {
                "accuracy": fold_summary["accuracy"]["mean"],  # type: ignore[index]
                "per_class": {
                    class_name: {
                        metric: fold_summary["per_class"][class_name][metric]["mean"]  # type: ignore[index]
                        for metric in ("precision", "recall", "f1")
                    }
                    for class_name in CLASS_NAMES
                },
            }
        )

    report = {
        "protocol": (
            "leave-one-author-out"
            if args.leave_one_author_out
            else "single-author-holdout"
        ),
        "authors": sorted({str(author) for author in authors}),
        "test_authors": test_authors,
        "classes": list(CLASS_NAMES),
        "split_seed": args.split_seed,
        "seeds": list(args.seeds),
        "folds": fold_results,
        "out_of_fold_runs": out_of_fold_runs,
        "out_of_fold_summary_across_seeds": summarize_runs(out_of_fold_runs),
        "summary_across_authors": (
            summarize_runs(fold_mean_runs) if len(fold_mean_runs) > 1 else None
        ),
        "confidence_interval_note": (
            "Los IC95% entre semillas resumen variación de entrenamiento. "
            "El resumen entre autores es exploratorio y usa solo los autores presentes; "
            "ninguno de estos intervalos sustituye una prueba externa con una quinta persona. "
            "Los límites se restringen al rango válido [0, 1]."
        ),
    }
    with (protocol_dir / "evaluation.json").open("w", encoding="utf-8") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
    print(f"Resultados guardados en: {protocol_dir.resolve()}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("."))
    parser.add_argument("--output-dir", type=Path, default=Path("model_results"))
    test_mode = parser.add_mutually_exclusive_group()
    test_mode.add_argument("--test-author")
    test_mode.add_argument(
        "--leave-one-author-out",
        action="store_true",
        help="Ejecuta un fold por cada autor presente en autores.npy.",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=list(DEFAULT_SEEDS))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=6)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--cam-per-class", type=int, default=2)
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1 or args.patience < 1:
        parser.error("epochs, batch-size y patience deben ser mayores que cero")
    if not args.seeds:
        parser.error("indica al menos una semilla")
    return args


if __name__ == "__main__":
    run(parse_args())
