import pandas as pd
import datetime

def _find_column_loc(col,df):
    column_position = df.columns.get_loc(col)
    return column_position


def _find_ue_after_ce_storm(ce_storm,ue):
    ue_after_ce=ue[(ue["TimeStamp"]>ce_storm["TimeStamp"])&(ue["Server"]==ce_storm["Server"])]
    print(len(ue_after_ce))
    return len(ue_after_ce)


def merge_and_filter_unmatched(df, ce_storm_df, threshold):
    no_storm = df[df["Machin_Level_DeltCE"] < threshold]
    no_storm = no_storm.drop_duplicates(subset=["Server", "Datacenter"], keep='first')
    merged_df = pd.merge(no_storm, ce_storm_df[['Server', 'Datacenter']], on=['Server', 'Datacenter'], how='left', indicator=True)
    no_storm_filtered = merged_df[merged_df['_merge'] == 'left_only'].drop('_merge', axis=1)
    no_storm_filtered = no_storm_filtered.drop_duplicates()
    no_storm_filtered = no_storm_filtered.reset_index(drop=True)
    unmatched_rows = df[~df.index.isin(no_storm_filtered.index)]
    return no_storm_filtered, unmatched_rows


def calculate_machine(df,uedf):
    df = df.drop_duplicates(subset=["Server", "Datacenter"], keep='first')
    df["UETime"] = 0
    loc = _find_column_loc("UETime", df)
    for i in range(len(df)):
        result_df = uedf[(uedf["Server"] == df.iloc[i]["Server"]) & (uedf["Datacenter"] == df.iloc[i]["Datacenter"]) & (
                uedf["TimeStamp"] > df.iloc[i]["TimeStamp"])]
        if result_df.empty:
            continue
        else:
            df.iloc[i, loc] = result_df.iloc[0]["TimeStamp"]
    return df



def calculate_no_storm_machine(df,uedf):
    df = df.drop_duplicates(subset=["Server", "Datacenter"], keep='first')
    df["UETime"] = 0
    loc = _find_column_loc("UETime", df)
    for i in range(len(df)):
        result_df = uedf[(uedf["Server"] == df.iloc[i]["Server"]) & (uedf["Datacenter"] == df.iloc[i]["Datacenter"])]
        if result_df.empty:
            continue
        else:
            df.iloc[i, loc] = result_df.iloc[0]["TimeStamp"]
    return df


def ce_storm_analysis(df,uedf,threshold):
    uedf = uedf.sort_values(by="TimeStamp")
    df = df.sort_values(by="TimeStamp")
    #  Servers with CE storms
    ce_storm_df = df[df["Machin_Level_DeltCE"] >= threshold]
    ce_storm_df=calculate_machine(ce_storm_df,uedf)
    ce_storm_machine=len(ce_storm_df)
    uer_machine=len(ce_storm_df[ce_storm_df["UETime"]!=0])
    ce_storm_machine_ratio=uer_machine/ce_storm_machine
    # Servers with no CE storm
    no_ce_storm_df, unmatch = merge_and_filter_unmatched(df, ce_storm_df, threshold)
    no_ce_storm_df = calculate_no_storm_machine(no_ce_storm_df, uedf)
    no_ce_storm_machine=len(no_ce_storm_df)
    no_ce_storm_uer_machine = len(no_ce_storm_df[no_ce_storm_df["UETime"] != 0])
    no_ce_storm_machine_ratio=no_ce_storm_uer_machine/no_ce_storm_machine
    return ce_storm_machine_ratio,no_ce_storm_machine_ratio


path= "data/delta_ce.csv"
path1="data/HBM_all_ue.csv"
df=pd.read_csv(path)
uedf=pd.read_csv(path1)
threshold=[10,20,30]
print("========================CE storm analysis========================")
for value in threshold:
    print(f"=============Threshold {value}=============")
    ce_storm_machine_ratio,no_ce_storm_machine_ratio=ce_storm_analysis(df,uedf,value)
    print("The ration of servers with CE storms:",ce_storm_machine_ratio)
    print("The ration of servers with no CE storm:",no_ce_storm_machine_ratio)
