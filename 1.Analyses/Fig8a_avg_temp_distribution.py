# encoding: utf-8
import os
import pandas as pd

list_ue = []

list_ue_1d_avg = []

cnt_ue = 0
list_ce = []

list_ce_1d_avg = []

cnt_ce = 0

for i in range(0,8):

    list_ce_1d_avg.append(0)
    list_ue_1d_avg.append(0)


def ue_get_temp_ratio(file_path):
    csv_single_df = pd.read_csv(file_path)
    checked_df = csv_single_df[(csv_single_df["1h_avg"] > 0) & (csv_single_df["7d_max"] <= 200)]
    global list_ue_1h_avg
    global list_ue_3h_avg
    global list_ue_1d_avg
    global list_ue_7d_avg
    global cnt_ue
    for i in range(0,8):
        if i == 0:
            list_ue_1d_avg[i] =len(checked_df[(checked_df["1d_avg"] < i + 25)])
        elif i == 7:
            list_ue_1d_avg[i] =len(checked_df[(checked_df["1d_avg"] >= 55)])
        else:
            list_ue_1d_avg[i] =len(checked_df[(checked_df["1d_avg"] >= 5*i + 20) & (checked_df["1d_avg"] < 5*i + 25)])
    cnt_ue = len(checked_df)


def ce_get_temp_ratio(file_path):
    csv_single_df = pd.read_csv(file_path)
    checked_df = csv_single_df[(csv_single_df["1h_avg"] > 0) & (csv_single_df["7d_max"] <= 200)]
    global list_ce_1h_avg
    global list_ce_3h_avg
    global list_ce_1d_avg
    global list_ce_7d_avg
    global cnt_ce
    for i in range(0,8):
        if i == 0:
            list_ce_1d_avg[i] =len(checked_df[(checked_df["1d_avg"] < i + 25)])

        elif i == 7:
            list_ce_1d_avg[i] =len(checked_df[(checked_df["1d_avg"] >= 55)])
        else:

            list_ce_1d_avg[i] =len(checked_df[(checked_df["1d_avg"] >= 5*i + 20) & (checked_df["1d_avg"] < 5*i + 25)])

    cnt_ce = len(checked_df)


def print_file_contents(file_path):
    with open(file_path, 'r') as file:
        print(file.read())


in_path1 = "./data/uer_temp.csv"
in_path2 = "./data/ce_temp.csv"
out_file1 = "./result/avg_ue_temp_distribution.txt"
out_file2 = "./result/avg_ce_temp_distribution.txt"

ue_get_temp_ratio(in_path1)
ce_get_temp_ratio(in_path2)

r_f1 = open(out_file1,'w')
r_f2 = open(out_file2,'w')
# r_f1.write("UE总数:  "+str(cnt_ue))
# r_f2.write("CE总数:  "+str(cnt_ce))

for i in range(0,8):
    if i == 0:
        str_init = "<25 :"
        r_f1.write(str_init)
        r_f2.write(str_init)

        ue_1d_cnt = list_ue_1d_avg[i]

        str_ue_list_1d =  "    Ratio:" + str(round(float(ue_1d_cnt) / cnt_ue, 4))+"\n"

        ce_1d_cnt = list_ce_1d_avg[i]

        str_ce_list_1d =  "    Ratio:" + str(round(float(ce_1d_cnt) / cnt_ce, 4))+"\n"


        r_f1.write(str_ue_list_1d)
        r_f2.write(str_ce_list_1d)


    elif i == 7:
        str_init = ">55 :"
        r_f1.write(str_init)
        r_f2.write(str_init)

        ue_1d_cnt = list_ue_1d_avg[i]

        str_ue_list_1d =  "    Ratio:" + str(round(float(ue_1d_cnt) / cnt_ue, 4)) + "\n"
        ce_1d_cnt = list_ce_1d_avg[i]


        str_ce_list_1d =  "    Ratio:" + str(round(float(ce_1d_cnt) / cnt_ce, 4)) + "\n"


        r_f1.write(str_ue_list_1d)
        r_f2.write(str_ce_list_1d)

    else:
        str_init = "["+str(5*i+20)+","+str(5*i+25)+") :"
        r_f1.write(str_init)
        r_f2.write(str_init)

        ue_1d_cnt = list_ue_1d_avg[i]

        str_ue_list_1d =  "    Ratio:" + str(round(float(ue_1d_cnt) / cnt_ue, 4)) + "\n"

        ce_1d_cnt = list_ce_1d_avg[i]

        str_ce_list_1d =  "    Ratio:" + str(round(float(ce_1d_cnt) / cnt_ce, 4)) + "\n"


        r_f1.write(str_ue_list_1d)
        r_f2.write(str_ce_list_1d)
r_f1.close()
r_f2.close()
# Print the contents of the output files
print("===================== Average Temperature Distribution =====================")
print("***************************************")
print("==========CE Average Temperature Distribution ==========")
print_file_contents(out_file2)
print("***************************************")
print("==========UE Average Temperature Distribution==========")
print_file_contents(out_file1)


# out_list1 = "./ue_temp_avg"
# out_list2 = "./ce_temp_avg"
#
# li_1 = []
# li_2 = []
# for i in range(0,8):
#     x1 = round(float(list_ue_1d_avg[i])/cnt_ue,4)
#     li_1.append(x1)
#     x2 = round(float(list_ce_1d_avg[i])/cnt_ce,4)
#     li_2.append(x2)
#
# s1 ="["
# s2 ="["
# for i in range(0,8):
#     if i != 7:
#         s1 = s1 + str(li_1[i])+","
#         s2 = s2 + str(li_2[i])+","
#     else:
#         s1 = s1 +str(li_1[i])
#         s2 = s2 +str(li_2[i])
# s1 =s1 +"]"
# s2 = s2+"]"
#
# l_f1 = open(out_list1,'w')
# l_f2 = open(out_list2,'w')
# l_f1.write(s1)
# l_f2.write(s2)