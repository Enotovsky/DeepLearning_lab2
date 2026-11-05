"""10 разных фотографий без изменения разрешения. Старый main.py сохранён."""
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from PIL import Image
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parent
# Файл, правильный класс ImageNet, визуальная крупность объекта в кадре.
# «Близко/далеко» — оценка по кадру, расстояние до камеры неизвестно.
IMAGES = [
    ('fish_near', 0, 'Крупный план'),
    ('fish_far', 0, 'Общий план'),
    ('dog_near', 217, 'Крупный план'),
    ('dog_far', 217, 'Общий план'),
    ('church_near', 497, 'Крупный план'),
    ('church_far', 497, 'Общий план'),
    ('truck_near', 569, 'Крупный план'),
    ('truck_far', 569, 'Общий план'),
    ('parachute_near', 701, 'Крупный план'),
    ('parachute_far', 701, 'Общий план'),
]


def main():
    torch.set_num_threads(4)
    torch.hub.set_dir(str(ROOT / 'data' / 'torch'))
    # Только тензор и нормализация: нет Resize, Crop, padding и изменения масштаба.
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    print('Загрузка двух сетей...', flush=True)
    networks = {
        'ResNet18': models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).eval(),
        'ResNet152': models.resnet152(weights=models.ResNet152_Weights.IMAGENET1K_V1).eval(),
    }
    labels = models.ResNet18_Weights.IMAGENET1K_V1.meta['categories']
    results = []
    print('ResNet18 и ResNet152: 10 исходных фотографий')
    print('Пять классов, по два разных снимка с разной крупностью объекта.')
    print('Размеры файлов из набора Imagenette160 сохранены. Программа не меняет разрешение.')
    print('Крупность оценена визуально; фактическое расстояние до камеры неизвестно.')
    print('Фото разные: помимо масштаба отличаются фон, ракурс и освещение.')
    with torch.inference_mode():
        # Прогрев на первом исходном снимке, без изменения его размеров.
        with Image.open(ROOT / 'data_original' / 'fish_near.jpeg') as source:
            warmup = preprocess(source.convert('RGB')).unsqueeze(0)
        for model in networks.values():
            model(warmup)

        for number, (filename, target, plan) in enumerate(IMAGES, 1):
            with Image.open(ROOT / 'data_original' / f'{filename}.jpeg') as source:
                picture = source.convert('RGB')
            tensor = preprocess(picture).unsqueeze(0)
            # Проверка: высота и ширина входа совпадают с исходным файлом.
            assert tuple(tensor.shape[-2:]) == (picture.height, picture.width)
            print(f'\n{number}. {labels[target]} — {plan}, {picture.width}x{picture.height}')
            print('| Сеть | Ответ | Уверенность | Верно | Время, мс |')
            captions = []
            for name, model in networks.items():
                start = time.perf_counter()
                scores = model(tensor)
                elapsed = (time.perf_counter() - start) * 1000
                probabilities = scores.softmax(dim=1)[0]
                prediction = probabilities.argmax().item()
                confidence = probabilities[prediction].item()
                correct = prediction == target
                answer = 'да' if correct else 'нет'
                print(f'| {name} | {labels[prediction]} | {confidence:.1%} | {answer} | {elapsed:.1f} |')
                results.append((name, plan, correct, elapsed))
                captions.append(f'{name}: {labels[prediction]}, {confidence:.1%}, верно: {answer}')

            fig, ax = plt.subplots(figsize=(9, 7))
            # Отображение на экране не меняет тензор, поданный в сеть.
            ax.imshow(picture, interpolation='nearest')
            ax.set_title(f'{number}/10. {labels[target]} — {plan}\n'
                         f'Исходный размер: {picture.width}x{picture.height}\n' + '\n'.join(captions))
            ax.axis('off')
            fig.tight_layout()
            plt.show(block=True)
            plt.close(fig)

    print('\nИтоговое сравнение')
    print('| Сеть | План | Верно | Точность | Среднее время, мс |')
    for name in networks:
        for plan in ['Крупный план', 'Общий план', 'Все']:
            group = [r for r in results if r[0] == name and (plan == 'Все' or r[1] == plan)]
            count = sum(r[2] for r in group)
            print(f'| {name} | {plan} | {count}/{len(group)} | {count / len(group):.0%} | '
                 f'{sum(r[3] for r in group) / len(group):.1f} |')
    print('\nУверенность — вероятность выбранного класса, а не точность сети.')
    print('Время — один вызов сети после прогрева, без чтения файла, обработки и показа.')
    print('Размер входа различается, поэтому на время влияют и архитектура, и разрешение.')
    print('Всего пять примеров на план: по ним нельзя выделить влияние только расстояния.')


if __name__ == '__main__':
    main()
