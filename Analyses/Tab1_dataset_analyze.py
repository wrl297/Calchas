import pandas as pd
# Dataset Overview
def find_data_detail(df,error_type):
    # HBM level
    HBM_number=df.drop_duplicates(
        subset=["Datacenter","Server", "Name", "Stack"],
        keep='first')
    print(error_type+"HBM:",len(HBM_number))
    # SID level
    SID_number=df.drop_duplicates(
        subset=["Datacenter","Server", "Name", "Stack",  "SID"],
        keep='first')
    print(error_type + "SID:", len(SID_number))
    # PS-CH level
    PS_CH_number=_number=df.drop_duplicates(
        subset=["Datacenter","Server", "Name", "Stack", "PcId", "SID"],
        keep='first')
    print(error_type + "PS-CH:", len( PS_CH_number))
    # BankGroup
    BG_number=df.drop_duplicates(
        subset=["Datacenter","Server", "Name", "Stack", "PcId", "SID", "BankGroup"],
        keep='first')
    print(error_type+ "BG:",  len(BG_number))
    # Bank
    Bank_number=df.drop_duplicates(
        subset=["Datacenter","Server", "Name", "Stack", "PcId", "SID", "BankArray", "BankGroup"],
        keep='first')
    print(error_type + "Bank:", len(Bank_number))
    # Row
    Row_number=df.drop_duplicates(
        subset=["Server", "Name", "Stack", "PcId", "SID", "BankArray", "BankGroup", "Datacenter", "Row"],
        keep='first')
    print(error_type + "Row:", len(Row_number))
    # Col
    Col_number=df.drop_duplicates(
        subset=["Server", "Name", "Stack", "PcId", "SID", "BankArray", "BankGroup", "Datacenter", "Col"],
        keep='first')
    print(error_type + "Col:", len(Col_number))
    # Cell
    Cell_number = df.drop_duplicates(
        subset=["Server", "Name", "Stack", "PcId", "SID", "BankArray", "BankGroup", "Datacenter", "Row","Col"],
        keep='first')
    print(error_type + "Cell:", len(Cell_number))


# Count total number of different errors
def error_number_of_device(path,path1):
    print("*********")
    print("========Total number of different errors========")
    dfce = pd.read_csv(path1)
    dfce = dfce.fillna('no')
    df1 = dfce[dfce['EccType'] == 'SingleBitEcc']
    df2 = df1[["Server", "Name", "Position", "EccType", "Stack", "PcId", "SID", "BankArray", "BankGroup",
               "Row", "Col", "ErrorCount"]]
    dfstor = df2.drop_duplicates(
        subset=["Server", "Name",  "Position", "EccType", "Stack", "PcId", "SID", "BankArray",
                "BankGroup", "Row", "Col"], keep='first')  # 去重
    i = 0
    CE_number = 0
    dfstor = dfstor[dfstor["ErrorCount"] < 100000]
    while i < len(dfstor):
        dfx = dfstor.iloc[i]
        df3 = df1[(df1["Server"] == dfx["Server"]) & (df1["Name"] == dfx["Name"]) & (df1["Position"] == dfx["Position"]) & (
                    df1["Col"] == dfx["Col"]) & (df1["Row"] == dfx["Row"])]["ErrorCount"]
        maxA = df3.max(axis=0)
        CE_number = maxA + CE_number
        i = i + 1
    print("CE Number:", CE_number)
    df = pd.read_csv(path)
    dfueo = df[df["EccType"] == "UEO"]
    print("UEO Number:",len(dfueo))
    dfu = df[df["EccType"] == "UER"]
    print("UER Number:",len(dfu))


# Count number of device levels with errors
def device_level_with_error(path):
    df = pd.read_csv(path)
    print("*********")
    print("========Number of device levels with errors========")
    dfce = df[df["EccType"] == "CE"]
    print("========CE========")
    dfce = dfce.drop_duplicates(
        subset=["Server", "Name", "Stack", "PcId", "SID", "BankArray", "BankGroup", "Datacenter", "Col", "Row", "TimeStamp"],
        keep='first')
    find_data_detail(dfce, "CE_")
    print("========UEO========")
    dfueo = df[df["EccType"] == "UEO"]
    dfueo = dfueo.drop_duplicates(
        subset=["Server", "Name", "Stack", "PcId", "SID", "BankArray", "BankGroup", "Datacenter", "Col", "Row", "TimeStamp"],
        keep='first')
    find_data_detail(dfueo, "UEO_")
    print("========UER========")
    dfu = df[df["EccType"] == "UER"]
    dfu = dfu.drop_duplicates(
        subset=["Server", "Name", "Stack", "PcId", "SID", "BankArray", "BankGroup", "Datacenter", "Col", "Row", "TimeStamp"],
        keep='first')
    find_data_detail(dfu, "UER_")



path= "data/fault_data_predict.csv"
path1="data/dataFinal.csv"
print("========================Dataset Overview========================")
error_number_of_device(path,path1)
device_level_with_error(path)



