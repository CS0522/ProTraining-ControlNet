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
output_file_path = "./training/danbooruSmall/prompt.json"
# open file path
open_file_path = "./training/danbooruSmall/prompts/"
# pictures file path
pic_file_path = "./training/danbooruSmall/source/"

"""
遍历 prompts 文件夹内文件
因为文件名相同，所以直接获取文件名（除去后缀）即可
然后写入 output_file 文件中
"""
def get_prompt():
    # open the output temp file
    output_file = open(output_file_path, 'w', encoding = 'utf-8')

    # 获取文件夹中存在的 pic 的文件名（共 10000 个）
    pic_filenames = []
    for pic_file in os.listdir(pic_file_path):
        (pic_filename, file_extension_name) = os.path.splitext(pic_file)
        pic_filenames.append(pic_filename)
    # print(pic_filenames)
    # print(len(pic_filenames))

    for pic_filename in pic_filenames:
        # open prompt file
        prompt_file = open(open_file_path + pic_filename + ".txt")
        # get the one line of the file
        line = prompt_file.readline()
        # 去除逗号，去除反斜杠，去除 ""
        # line = line.replace(",", "")
        line = line.replace("\\", "")
        line = line.replace("\"", "\'")
        # print(open_file_path + file + line)
        # write into the output file in format
        output_file.write(concat1 + pic_filename + concat2 + pic_filename + concat3 + line + concat4 + "\n")
        # close 
        prompt_file.close()

    
    # 循环遍历
    # for file in os.listdir(open_file_path):
    #     (filename,file_extension_name) = os.path.splitext(file)
        
    #     # 判断当前的 filename 是否在 pic_filenames 中已存在
    #     if (filename not in pic_filenames):
    #         continue

    #     # open the current file
    #     current_file = open(open_file_path + file)
    #     # get the one line of the file
    #     line = current_file.readline()
    #     # 去除逗号，去除反斜杠，去除 ""
    #     # line = line.replace(",", "")
    #     line = line.replace("\\", "")
    #     line = line.replace("\"", "\'")
    #     # print(open_file_path + file + line)
    #     # write into the output file in format
    #     output_file.write(concat1 + filename + concat2 + filename + concat3 + line + concat4 + "\n")
    #     # close 
    #     current_file.close()

    # close 
    output_file.close()

get_prompt()