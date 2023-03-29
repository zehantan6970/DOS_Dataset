import json
import os
#统计包含特定实例数的图片数量
def main(args):
    object={}
    files = os.listdir(args.input_dir)
    for file in files:
        with open(args.input_dir+'/'+file,'r') as f:
            data = json.load(f)
            num = len(data['shapes'])
            num = str(num)
            if num in object:
                object[num]+=1
            else:
                object[num]=1
    print(object)

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_dir", default='./val_annotations', type=str,
                        help="input annotation directory")
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    main(args)
