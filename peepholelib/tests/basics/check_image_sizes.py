from pathlib import Path
from PIL import Image
from collections import Counter

IMG_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp'}

if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print('Usage: python check_image_sizes.py <dataset_path> [<dataset_path> ...]')
        sys.exit(1)
    for path in sys.argv[1:]:
        root = Path(path)
        sizes = []
        for f in root.rglob('*'):
            if f.suffix.lower() in IMG_EXTENSIONS:
                try:
                    sizes.append(Image.open(f).size)  # (width, height)
                except Exception:
                    pass

        if not sizes:
            print(f'{root.name}: no images found')
            break

        widths  = [s[0] for s in sizes]
        heights = [s[1] for s in sizes]
        unique  = Counter(sizes)

        print(f'\n{root.name} ({len(sizes)} images)')
        print(f'  width  — min: {min(widths)}, max: {max(widths)}, avg: {sum(widths)/len(widths):.1f}')
        print(f'  height — min: {min(heights)}, max: {max(heights)}, avg: {sum(heights)/len(heights):.1f}')
        print(f'  unique sizes: {len(unique)}')
