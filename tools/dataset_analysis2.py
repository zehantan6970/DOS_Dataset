import json
import math

#统计图片分辨率、实例数、各类别实例数量、各实例标注区域面积
def main(args):
    shapes = {}
    nums = 0
    category = {}
    areas = {'1': {'0-16': 0, '16-32': 0, '32-64': 0, '64-128': 0, '128-256': 0, '256-512':   0,'512-1024':0,'1024-n':0},
         '2': {'0-16': 0, '16-32': 0, '32-64': 0, '64-128': 0, '128-256': 0, '256-512':0,'512-1024':0,'1024-n':0},
         '3': {'0-16': 0, '16-32': 0, '32-64': 0, '64-128': 0, '128-256': 0, '256-512':0,'512-1024':0,'1024-n':0},
         '4': {'0-16': 0, '16-32': 0, '32-64': 0, '64-128': 0, '128-256': 0, '256-512':0,'512-1024':0,'1024-n':0}}

    with open(args.path, 'r') as f:
        data = json.load(f)
        images = data['images']
        for image in images:
            shape = str(image['height']) + '_' + str(image['width'])
            if shape in shapes:
                shapes[shape] += 1
            else:
                shapes[shape] = 1

        annotations = data['annotations']
        for annotation in annotations:
            nums += 1
            id = str(annotation['category_id'])
            if id in category:
                category[id] += 1
            else:
                category[id] = 1

            area = annotation['area']
            area = math.sqrt(area)
            if area <= 16:
                areas[id]['0-16'] += 1
            elif 16 < area <= 32:
                areas[id]['16-32'] += 1
            elif 32 < area <= 64:
                areas[id]['32-64'] += 1
            elif 64 < area <= 128:
                areas[id]['64-128'] += 1
            elif 128 < area <= 256:
                areas[id]['128-256'] += 1
            elif 256 < area <= 512:
                areas[id]['256-512'] += 1
            elif 512 < area <= 1024:
                areas[id]['512-1024'] += 1
            else:
                areas[id]['1024-n'] += 1

    print(nums)
    print(shapes)
    print(category)
    print(areas)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", default='./train_coco/annotations.json', type=str,
                        help="path to coco annotations.json")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
