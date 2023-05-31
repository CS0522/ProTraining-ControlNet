import os

# {"source": "source/0.png", "target": "target/0.png", "prompt": "pale golden rod circle with old lace background"}
# source: 线稿图（sketch）
# target: 原图（src）

# 需要拼接的语句
concat1 = "{\"source\": \"source/"
concat2 = ".png\", \"target\": \"target/"
concat3 = ".png\", \"prompt\": \""
concat4 = "\"}"

# output file 
output_file_path = "./prompt.json"
# open file path
open_file_path = "./prompts/"

"""
遍历 prompts 文件夹内文件
因为文件名相同，所以直接获取文件名（除去后缀）即可
然后写入 output_file 文件中
"""
def get_prompt():
    # open the output temp file
    output_file = open(output_file_path, 'w', encoding = 'utf-8')
    # 循环遍历
    for file in os.listdir(open_file_path):
        (filename,file_extension_name) = os.path.splitext(file)
        # open the current file
        current_file = open(open_file_path + file)
        # get the one line of the file
        line = current_file.readline()
        # 去除逗号，去除反斜杠
        # line = line.replace(",", "")
        line = line.replace("\\", "")
        # print(open_file_path + file + line)
        # write into the output file in format
        output_file.write(concat1 + filename + concat2 + filename + concat3 + line + concat4 + "\n")
        # close 
        current_file.close()

    # close 
    output_file.close()

get_prompt()