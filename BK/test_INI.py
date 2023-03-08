from configparser import ConfigParser

parser = ConfigParser()
parser.read('setting.ini')

read_path = parser.get('path', 'read_path')  # 
result_path = parser.get('path', 'result_path')  #

count = parser.get('top-k', 'count')  #
score_threshold = parser.get('top-k', 'score_threshold')  #

print(read_path)
print(result_path)

print(count)
print(score_threshold)

# print(type(count))
# print(type(score_threshold))


'''
config = ConfigParser()
config['path'] = {
    'read_path': 'D:/2022/3.YOLACT/data',
    'result_path': 'D:/2022/3.YOLACT/output'
}

config['top-k'] = {
    'count': '2',
    'score_threshold': '0.5'
}



#
# config['db'] = {
#     'db_name': 'myapp_dev',
#     'db_host': 'localhost',
#     'db_port': '8889'
# }
#
# config['files'] = {
#     'use_cdn': 'false',
#     'images_path': '/my_app/images'
# }
with open('./setting.ini', 'w') as f:
      config.write(f)
'''

