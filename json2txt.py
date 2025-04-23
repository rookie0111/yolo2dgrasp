import os
import json
from pathlib import Path


def convert_json_to_txt(input_folder, output_folder):
    # 创建输出文件夹
    Path(output_folder).mkdir(parents=True, exist_ok=True)

    # 遍历输入文件夹中的所有JSON文件
    for filename in os.listdir(input_folder):
        if not filename.endswith('.json'):
            continue

        input_path = os.path.join(input_folder, filename)
        output_filename = filename.replace('.json', '.txt')
        output_path = os.path.join(output_folder, output_filename)

        try:
            with open(input_path, 'r', encoding='utf-8') as f:
                data = json.load(f)

            lines = []
            for shape in data['shapes']:
                # 提取坐标点并展平为列表
                points = [str(coord) for point in shape['points'] for coord in point]
                # 转换direction为整数
                direction = str(int(shape['direction']))
                # 构建行内容
                line = ' '.join(points) + f' {shape["label"]} {direction}\n'
                lines.append(line)

            # 写入TXT文件
            with open(output_path, 'w', encoding='utf-8') as f:
                f.writelines(lines)

            print(f'成功转换: {filename} -> {output_filename}')

        except Exception as e:
            print(f'处理文件 {filename} 时出错: {str(e)}')


if __name__ == '__main__':
    # 配置路径
    input_folder = r'D:\edgedownoad\3\3'  # 替换为你的JSON文件夹路径
    output_folder = r'D:\edgedownoad\3\3\labels'  # 输出文件夹名称

    # 执行转换
    convert_json_to_txt(input_folder, output_folder)
    print('所有文件转换完成！')