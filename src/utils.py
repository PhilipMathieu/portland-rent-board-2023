"""

@author Philip Mathieu

This file contains functions for cleaning and analyzing Long Term Rental data.
The main function is get_data(), which returns a cleaned dataframe of LTR data.
The other functions are used internally by get_data(), but can be used on their own.

The add_ward() and add_ward_geo() functions require the "parcels_ward_gis.csv" and
"parcels_ward_gis.geojson" files, respectively. These files contain information identifying
the voting ward that each parcel belongs to. Because this is considered voter information, it
must be obtained directly from the City Clerk's office.

Example Usage:
    import utils
    df = utils.get_data("2023)

    # get the number of units with a rent increase
    df[df["Rent_Inc"] > 0].shape[0]

    # get the number of units with a rent increase of 10% or more
    df[df["Rent_Inc_percent"] >= 10].shape[0]

    # get the number of units with a rent increase of 10% or more, excluding outliers
    df[(df["Rent_Inc_percent"] >= 10) & (~df["outlier"])].shape[0]
"""
import os
import numpy as np
import pandas as pd

numerical_columns = ["BaseRent1", "CurrentRent1", "PreviousRent", "BankedRent1", "Rent_Inc", 
                     'Rent_Inc_base', 'Rent_Inc_base_percent', 'Rent_Inc', 'Rent_Inc_per_BedRms', 
                     'Rent_Inc_percent', 'Rent_per_BedRms']

STREET_LIST_CSV = "../../municipal-street-list/parcels_ward_gis.csv"
STREET_LIST_GEOJSON = "../../municipal-street-list/parcels_ward_gis.geojson"

def int0(x):
    try:
        return int(x)
    except:
        return 0

def float0(x):
    try:
        return float(x)
    except:
        return 0
    
def dollars(x):
    x = x.replace("$", "").replace(",", "")
    return float0(x)

converters = {
    'TAXYEAR1':int,
    'LICENSENUMBER':str,
    'ADDRESS':str,
    'PARCELNUMBER':str,
    'NumberOfRentalUnits':int,
    'SVALUE1':str,
    'UPCSQualifiedUnits':int,
    'HQSQualifiedUnits':int,
    'Textbox50':int,
    'FullySprinkledBuildingQualifiedUnits':int,
    'BLDiscountAverageUnit':dollars,
    'ROWNUMBER1':int,
    'unitNumber1':str,
    'BaseRent1':dollars,
    'PreviousRent':dollars,
    'CurrentRent1':dollars,
    'BankedRent1':dollars,
    'CurrentSecurityDeposit1':dollars,
    'OtherPayments1':dollars,
    'nbrBedRms1':int0,
    'nbrBthRms1':float0,
    'kitInc1':str,
    'unitDesc2':str
}

def add_exempt(df, stats=False):
    df["exempt"] = (df["unitDesc2"] != 'None of the above') & (df["unitDesc2"] != '(Nothing Selected)')

    if stats:
        print("Breakdown by Exemption:")
        for val in df["unitDesc2"].unique():
            print(f"\t{val}: {df[df['unitDesc2'] == val].shape[0]}")
        N = df.shape[0]
        exempt = df["exempt"].sum()
        print(f"Total Exempt: {exempt} ({exempt / N * 100:2.0f}%)")
        print(f"Total Not Exempt: {N - exempt} ({(N - exempt) / N * 100:2.0f}%)")
    return df

def add_increases(df):
    df["Rent_Inc_base"] = df["CurrentRent1"] - df["BaseRent1"]
    df["Rent_Inc_base_percent"] = df["Rent_Inc_base"] / df["BaseRent1"] * 100

    df["Rent_Inc"] = df["CurrentRent1"] - df["PreviousRent"]
    df["Rent_Inc_per_BedRms"] = df["Rent_Inc"] / df["nbrBedRms1"].replace(0, 1) # replace 0 with 1 to account for studios
    df["Rent_Inc_percent"] = df["Rent_Inc"] / df["PreviousRent"] * 100

    df["Rent_per_BedRms"] = df["CurrentRent1"] / df["nbrBedRms1"].replace(0, 1) # replace 0 with 1 to account for studios

    # deal with infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    return df

def group_bedrooms(df):
    # create a new column with grouped bedrooms
    # units with 0 bedrooms are counted as 1, and units with 5+ bedrooms are counted as 5
    # return type is str so that "5+" can be used as a category
    df["nbrBedRms_studio"] = df["nbrBedRms1"].replace(0, 1)
    df["nbrBedRms_grouped"] = df["nbrBedRms1"].replace(0, 1)
    df["nbrBedRms_grouped"] = df["nbrBedRms_grouped"].where(df["nbrBedRms_grouped"] < 5, 5)
    df["nbrBedRms_grouped"] = df["nbrBedRms_grouped"].astype(str)
    df["nbrBedRms_grouped"] = df["nbrBedRms_grouped"].replace("5", "5+")
    return df

def add_outlier_2022(df, stats=False):
    if not "Rent_Inc" in df.columns:
        df = add_increases(df)
    N = df.shape[0]

    # count $0 rents
    df["outlier_0_rent"] = df["CurrentRent1"] == 0
    N_0_rent = df["outlier_0_rent"].sum()

    # drop unrealistically low and high rents
    df["outlier_rent"] = (df["CurrentRent1"] <= 100) | (df["CurrentRent1"] >= 6500)
    N_rent = df["outlier_rent"].sum()
    # drop unrealistic increases/decreases vs base rent if provided
    df["outlier_inc_base"] = ((df["Rent_Inc_base_percent"] <= -10) | (df["Rent_Inc_base_percent"] >= 40)) & (df["BaseRent1"] != 0)
    N_inc_base = df["outlier_inc_base"].sum()
    # drop unrealistic rent increases
    df["outlier_inc_prev"] = (df["Rent_Inc_percent"] <= -10) | (df["Rent_Inc_percent"] >= 40)
    N_inc_prev = df["outlier_inc_prev"].sum()
    # check overall outlier count
    df["outlier"] = df["outlier_rent"] | df["outlier_inc_base"] | df["outlier_inc_prev"]
    N_outliers = df["outlier"].sum()
    
    if stats:
        print(f"Breakdown by Outlier Condition:")
        print(f"\tOutlier Rents ($0): {N_0_rent} ({N_0_rent/ N_outliers * 100:2.0f}%)")
        print(f"\tOutlier Rents (other): {N_rent - N_0_rent} ({(N_rent - N_0_rent)/ N_outliers * 100:2.0f}%)")
        print(f"\tOutlier Increase vs Base: {N_inc_base} ({N_inc_base / N_outliers * 100:2.0f}%)")
        print(f"\tOutlier Increase vs Previous: {N_inc_prev} ({N_inc_prev / N_outliers * 100:2.0f}%)")
        print(f"\tOverall: {N_outliers} ({N_outliers / N_outliers* 100:2.0f}%)")

    return df

def add_outlier_2023(df, stats=False):
    if not "Rent_Inc" in df.columns:
        df = add_increases(df)
    N = df.shape[0]

    # drop units with 0 rent
    df["outlier_0_rent"] = df["CurrentRent1"] == 0
    N_0_rent = df["outlier_0_rent"].sum()
    # drop unrealistically low and high rents
    df["outlier_rent"] = ((df["CurrentRent1"] > 0) & (df["CurrentRent1"] <= 250)) | (df["CurrentRent1"] >= 4000)
    N_rent = df["outlier_rent"].sum()
    # drop unrealistic rent increases
    df["outlier_inc_prev"] = (df["Rent_Inc_percent"] <= -15) | (df["Rent_Inc_percent"] >= 65)
    N_inc_prev = df["outlier_inc_prev"].sum()
    # check overall outlier count
    df["outlier"] = df["outlier_0_rent"] | df["outlier_rent"] | df["outlier_inc_prev"]
    N_outliers = df["outlier"].sum()
    
    if stats:
        print(f"\nBreakdown by Outlier Condition:")
        print(f"\tOutlier $0 Rent: {N_0_rent} ({N_0_rent/ N * 100:2.0f}%)")
        print(f"\tOutlier Rents: {N_rent} ({N_rent/ N * 100:2.0f}%)")
        print(f"\tOutlier Increase vs Previous: {N_inc_prev} ({N_inc_prev / N * 100:2.0f}%)")
        print(f"\tOverall: {N_outliers} ({N_outliers / N * 100:2.0f}%)")

    return df


def add_ward(df):
    if "WARD" in df.columns:
        print("Warning: Overwriting Ward column")
    parcels = pd.read_file(STREET_LIST_CSV)
    parcels = parcels.sort_values("Ward_GIS").drop_duplicates("IAS_PARCEL_ID")
    merge = parcels.merge(df, right_on="PARCELNUMBER", left_on="IAS_PARCEL_ID", how="right")
    merge = merge.rename(columns={"Ward_GIS": "WARD"})
    merge["WARD_str"] = merge["WARD"].apply(lambda x: f"{x:.0f}")
    return merge

def add_ward_geo(df):
    try:
        import geopandas as gpd
    except:
        print("geopandas is not installed and is required for this function")
        return add_ward(df)
    
    if "WARD" in df.columns:
        print("Warning: Overwriting Ward column")
    parcels = gpd.read_file(STREET_LIST_GEOJSON).to_crs(3857)
    parcels = parcels.sort_values("Ward_GIS").drop_duplicates("IAS_PARCEL_ID")
    merge = parcels.merge(df, right_on="PARCELNUMBER", left_on="IAS_PARCEL_ID", how="right")
    merge = merge.rename(columns={"Ward_GIS": "WARD"})
    merge["WARD_str"] = merge["WARD"].apply(lambda x: f"{x:.0f}")
    return merge


def subset_stats(df):
    N = df.shape[0]
    N_no_outlier = df[~df["outlier"]].shape[0]
    outliers = N - N_no_outlier
    outliers_percent = outliers / N * 100
    N_rent_inc = df[df["Rent_Inc"] > 0].shape[0]
    N_rent_inc_percent = N_rent_inc / N * 100
    N_exempt = df[df["exempt"]].shape[0]
    N_exempt_percent = N_exempt / N * 100

    print("\nBreakdown by Subset:")
    print(f"\t{outliers} outliers ({outliers_percent:2.0f}%)")
    print(f"\t{N_no_outlier} non-outliers ({100-outliers_percent:2.0f}%)")
    print("\n")
    print(f"\t{N_rent_inc} rent increase ({N_rent_inc_percent:2.0f}%)")
    print(f"\t{N - N_rent_inc} no rent increase ({100 - N_rent_inc_percent:2.0f}%)")
    print("\n")
    print(f"\t{N_exempt} exempt ({N_exempt_percent:2.0f}%)")
    print(f"\t{N - N_exempt} not exempt ({100-N_exempt_percent:2.0f}%)")


def get_statistics(df):
    records = []
    for col in numerical_columns:
        record = {"column": col}

        # basic stats
        record["mean"] = df[col].mean()
        record["std"] = df[col].std()
        record["min"] = df[col].min()
        record["max"] = df[col].max()

        # quantiles
        record["1%"] = df[col].quantile(0.01)
        record["5%"] = df[col].quantile(0.05)
        record["25%"] = df[col].quantile(0.25)
        record["75%"] = df[col].quantile(0.75)
        record["95%"] = df[col].quantile(0.95)
        record["99%"] = df[col].quantile(0.99)

        # robust stats
        record["median"] = df[col].median()
        record["iqr"] = record["75%"] - record["25%"]
        record["lower"] = record["25%"] - 1.5 * record["iqr"]
        record["upper"] = record["75%"] + 1.5 * record["iqr"]

        # skip if df is a groupby object
        if not isinstance(df, pd.core.groupby.generic.DataFrameGroupBy):
            record["mad"] = abs(df[col] - record["median"]).median()
        records.append(record)
    return pd.DataFrame.from_records(records, index="column")
    
def get_data(filename_or_year=2023, outlier_method="2022", ward=False, geo=False):
    if os.path.isfile(filename_or_year):
        df = pd.read_csv(filename_or_year, converters=converters)
    else:
        df = pd.read_csv(f"../data/{str(filename_or_year)}-LTRs.csv", converters=converters)
    df["ID"] = df["LICENSENUMBER"].astype(str) + "-" + df["unitNumber1"].astype(str)
    df = add_exempt(df)
    df = add_increases(df)
    df = group_bedrooms(df)
    if outlier_method == "2022":
        df = add_outlier_2022(df, stats=True)
    elif outlier_method == "2023":
        df = add_outlier_2023(df, stats=True)
    if geo:
        df = add_ward_geo(df)
    elif ward:
        df = add_ward(df)
    df["Count"] = 1
    subset_stats(df)

    return df

