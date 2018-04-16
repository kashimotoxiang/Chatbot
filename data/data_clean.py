import os
import csv
import re

error_char = {r'，{1,}': r',',
              r'。{1,}': r'.',
              r'；{1,}': r';',
              r'：{1,}': r':',
              r'“{1,}': r'\'',
              r'”{1,}': r'\'',
              r'‘{1,}': r'\'',
              r'’{1,}': r'\'',
              r"！{1,}": r"!",
              r"？{1,}": r"?",
              r"＠{1,}": r"@",
              r"＿{1,}": r"_",
              r"（{1,}": r"(",
              r"\t{1,}": r" ",
              r"）{1,}": r")",
              r'\,{1,}': r',',
              r'\.{1,}': r'.',
              r'\;{1,}': r';',
              r'\:{1,}': r':',
              r'\'{1,}': r'\'',
              r'\'{1,}': r'\'',
              r'\'{1,}': r'\'',
              r'\'{1,}': r'\'',
              r"\!{1,}": r"!",
              r"\?{1,}": r"?",
              r"\@{1,}": r"@",
              r"\_{1,}": r"_",
              r"\({1,}": r"(",
              r"\){1,}": r")",
              #   r"健康咨询描述:": r"",
              #   r"想得到怎样的帮助:": r"",
              #   r"指导意见:": r"",
              #   r"病情分析:": r"",
              #   r"问题描述:": r"",
              #   r"发病时间:": r"",
              #   r"化验检查结果:": r"",
              #   r"想得到的帮助:": r"",
              #   r"曾经的治疗情况和效果:": r"",
              #   r"辅助检查:": r"",
              #   r"病史:": r"",
              #   r"其它:": r"",
              #   r"化验、检查结果:": r"",
              #   r"主要症状:": r"",
              #   r"检查所见情况:": r"",
              #   r"以往的诊断和治疗经过及效果:": r"",
              #   r"问题分析:": r"",
              }


result = []
with open('bigdata/csv_test.csv', 'r') as csvfile:
    csv_reader = csv.reader(csvfile)

    def reader(row):
        [file, questiontitle, questiondetails, accept_content] = row

        q = ''.join([questiontitle, questiondetails])
        a = accept_content

        def sample_process(data):
            for source, substitute in error_char.items():
                data = re.sub(source, substitute, data)
            return data
        q = sample_process(q)
        a = sample_process(a)
        result.append([q, a])
    list(map(reader, csv_reader))

with open('bigdata/aftercleaning_includesubtitle.csv', 'w') as csvfile:
    csv_reader = csv.writer(csvfile)
    for i, row in enumerate(result):
        csv_reader.writerow([i, row[0], row[1]])
