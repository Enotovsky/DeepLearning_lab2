"""Сравнение ResNet18 и ResNet152 на 10 фотографиях в трёх масштабах."""
import sys
import time
from pathlib import Path

import torch
from PIL import Image, ImageOps
from torchvision import models, transforms

# Русский текст в консоли Windows.
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent
# Имя снимка: номер правильного класса в ImageNet.
IMAGES = {
    'n01440764': 0,    # линь
    'n02102040': 217,  # английский спрингер-спаниель
    'n02979186': 482,  # кассетный проигрыватель
    'n03000684': 491,  # бензопила
    'n03028079': 497,  # церковь
    'n03394916': 566,  # валторна
    'n03417042': 569,  # мусоровоз
    'n03425413': 571,  # бензоколонка
    'n03445777': 574,  # мяч для гольфа
    'n03888257': 701,  # парашют
}
SCALES = {'Издалека': 0.3, 'Средний': 0.6, 'Вблизи': 1.0}


def change_scale(path, scale):
    """Меняем размер всей фотографии на одинаковом фоне 224x224."""
    with Image.open(path) as source:
        size = round(224 * scale)
        photo = ImageOps.contain(source.convert('RGB'), (size, size), Image.Resampling.LANCZOS)
    background = Image.new('RGB', (224, 224), (124, 116, 104))
    background.paste(photo, ((224 - photo.width) // 2, (224 - photo.height) // 2))
    return background


def main():
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT / 'data' / 'torch'))
    output = ROOT / 'results'
    output.mkdir(exist_ok=True)
    for image in IMAGES:
        if not (ROOT / 'data' / f'{image}.jpeg').is_file():
            raise FileNotFoundError(f'Нет data/{image}.jpeg. Сохраните папку data вместе с проектом.')

    # Подготовка пикселей так же, как при обучении на ImageNet.
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    networks = {
        'ResNet18': (models.resnet18, models.ResNet18_Weights.IMAGENET1K_V1),
        'ResNet152': (models.resnet152, models.ResNet152_Weights.IMAGENET1K_V1),
    }
    summary = []
    started = time.perf_counter()
    with (output / 'report.md').open('w', encoding='utf-8') as report:
        def show(text=''):
            # Один текст одновременно выводим в консоль и записываем в отчёт.
            print(text, flush=True)
            report.write(text + '\n')
            report.flush()

        show('# Сравнение ResNet18 и ResNet152 на CPU')
        show('10 фотографий Imagenette, три масштаба. Обе сети: ImageNet V1.')
        show('Масштаб имитируется размером фотографии на фоне 224x224, а не расстоянием камеры.')
        show('Уверенность — вероятность выбранного класса; она не гарантирует верный ответ.')
        for name, (build_model, weights) in networks.items():
            show(f'\n## {name}')
            model = build_model(weights=weights).eval()  # Режим распознавания.
            labels = weights.meta['categories']
            with torch.inference_mode():  # Градиенты не нужны: обучения нет.
                warmup = preprocess(change_scale(ROOT / 'data' / 'n01440764.jpeg', 1)).unsqueeze(0)
                model(warmup)  # Прогрев перед замером времени.
                for scale_name, scale in SCALES.items():
                    show(f'\n### {scale_name} ({scale:.0%})')
                    show('| Правильный класс | Ответ сети | Уверенность | Верно | мс |')
                    show('|---|---|---:|---|---:|')
                    correct, total_ms = 0, 0.0
                    for image, target in IMAGES.items():
                        picture = change_scale(ROOT / 'data' / f'{image}.jpeg', scale)
                        tensor = preprocess(picture).unsqueeze(0)  # [1, 3, 224, 224].
                        start = time.perf_counter()
                        scores = model(tensor)
                        elapsed = (time.perf_counter() - start) * 1000
                        probabilities = scores.softmax(dim=1)[0]
                        prediction = probabilities.argmax().item()
                        confidence = probabilities[prediction].item()
                        correct += int(prediction == target)
                        total_ms += elapsed
                        answer = 'да' if prediction == target else 'нет'
                        show(f'| {labels[target]} | {labels[prediction]} | {confidence:.1%} | {answer} | {elapsed:.1f} |')
                    show(f'Верно: {correct}/10 ({correct / 10:.0%}). Среднее время: {total_ms / 10:.1f} мс.')
                    summary.append((name, scale_name, correct, total_ms / 10))
            del model

        show('\n## Итоговое сравнение')
        show('| Модель | Масштаб | Верно из 10 | Точность | Среднее время, мс |')
        show('|---|---|---:|---:|---:|')
        for name, scale_name, correct, latency in summary:
            show(f'| {name} | {scale_name} | {correct} | {correct / 10:.0%} | {latency:.1f} |')
        show('\nТочность — доля верных ответов. Время измерено только для вызова сети, без обработки картинки.')
        show('На каждом масштабе 10 примеров: один ответ меняет точность на 10 процентных пунктов.')
        show('Сравните строки: какая сеть точнее, какая быстрее и как влияет масштаб.')
        show(f'\nПолное время выполнения: {time.perf_counter() - started:.1f} с.')
    print(f'\nОтчёт сохранён: {output / "report.md"}', flush=True)


if __name__ == '__main__':
    main()
