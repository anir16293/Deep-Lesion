import os

annotations_folder = '/Users/aniruddha/Downloads/deeplesion/Annotations'

xml_path = os.path.join('/Users/aniruddha/Downloads/deeplesion', 'xmllist.txt')

count = 0
with open(xml_path, 'w') as file:
    for file_name in os.listdir(annotations_folder):
        if '.xml' in file_name:
            #file_name = file_name.split('.')[0]
            file.write(file_name)
            file.write('\n')
            count += 1
print(count)

