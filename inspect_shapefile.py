import os
import geopandas
import pandas as pd
import math
from pprint import pprint

shapefile_path = os.path.join("doc-data", "195-ALTM-ALAP-lakes-withCentroid.shp")

shp_df = geopandas.read_file(shapefile_path)

shp_df = shp_df[
    [
        "OBJECTID",
        "NAME",
        "FTYPE",
        "FCODE",
        "FCODE_DESC",
        "SQKM",
        "SQMI",
        "Permanent_",
        "Resolution",
        "GNIS_ID",
        "GNIS_Name",
        "AreaSqKm",
        "Elevation",
        "ReachCode",
        "FType_2",
        "Shape_Area",
        "NHDPlusID",
        "area_ha",
        "Lon-Cent",
        "Lat-Cent",
    ]
]

# Site information

site_information = pd.read_excel(os.path.join("ALTM", "Site_Information_2022_8_1.xlsx"))

# filter so that program_id = LTM_ALTM
site_information = site_information[site_information["PROGRAM_ID"] == "LTM_ALTM"]

num_matches = 0
# now match shp file centroid to site information
for site_index, site_row in site_information.iterrows():  # O(n^2)
    acceptable_diff = 0.007

    for shp_index, shp_row in shp_df.iterrows():
        p1 = (site_row["LATDD_CENTROID"], site_row["LONDD_CENTROID"])  # lat, long
        p2 = (shp_row["Lat-Cent"], shp_row["Lon-Cent"])  # lat, long

        if math.dist(p1, p2) < acceptable_diff:
            site_id = site_row["SITE_ID"]

            print(
                f"{site_id}: Match between: SHP({shp_row["NAME"]}) and LTM_DATA({site_row["SITE_NAME"]}) | SHP_INDEX({shp_index}) | SHP_OBJECTID({shp_row["OBJECTID"]})"
            )
            num_matches += 1

            # shp_row["SITE_ID"] = site_id

            shp_df.at[shp_index, "SITE_ID"] = site_id  # modifies the original

print("# of matches: ", num_matches)


# ---------------------- Identify some good testing lakes ------------------------------------

lake_names_of_interest = [  # these match SHP file
    "Woods Lake",
    "Big Moose Lake",
    "Brook Trout Lake",
    "Dart Lake",
    "G Lake",
    "Indian Lake",  # Indian Lake is outlier, row 12 was the actual match, so drop rows with no siteid
    "Squaw Lake",
    "Moss Lake",
    "Otter Lake",
    "Queer Lake",
    "Raquette Lake Reservoir",
    "Sagamore Lake",
    "Cascade Lake",
    "Limekiln Lake",
    "North Lake",
    "Rondaxe, Lake",
    "South Lake",  # missing matched site_id
]

lake_infos_of_interest = []

for name_of_interest in lake_names_of_interest:
    matched_lakes_in_joined_shp_file = shp_df[shp_df["NAME"] == name_of_interest]

    matched_lakes_in_joined_shp_file = matched_lakes_in_joined_shp_file.dropna(
        subset=["SITE_ID"]
    )
    if len(matched_lakes_in_joined_shp_file) > 1:
        raise Exception("Multiple lakes in shapefile found with interested name")

    if len(matched_lakes_in_joined_shp_file) == 0:
        print(f'Lake of interest with name "{name_of_interest}" not found. ')
        continue

    lake_infos_of_interest.append(
        {
            "SITE_ID": matched_lakes_in_joined_shp_file.iloc[0]["SITE_ID"],
            "OBJECTID": matched_lakes_in_joined_shp_file.iloc[0]["OBJECTID"],
            "NAME": name_of_interest,
            "PERMANENT_ID": matched_lakes_in_joined_shp_file.iloc[0]["Permanent_"],
        }
    )

# -----------------------------------------------------------------------------------

truth_data = pd.read_excel(os.path.join("ALTM", "LTM_Data_2023_3_9.xlsx"))

truth_data = truth_data[
    ["SITE_ID", "DATE_SMP", "DOC_MG_L"]
]  # don't care abt other columns

# filter for after landsat 8 launch

truth_data = truth_data[truth_data["DATE_SMP"] > "2013-02-11"]

# Merge in shp file (object ids, centroids) into truth_data
truth_data = truth_data.merge(shp_df, left_on="SITE_ID", right_on="SITE_ID")

truth_data = truth_data[
    (truth_data["DATE_SMP"].dt.month > 4) & (truth_data["DATE_SMP"].dt.month < 11)
]

print("Truth data: \n", truth_data)

if __name__ == "__main__":
    # get mean doc for each lake

    mean_DOC_results = []

    for lake_info in lake_infos_of_interest:
        lake_name = lake_info["NAME"].lower().replace(" ", "_")
        lake_objectid = lake_info["OBJECTID"]

        # filter truth data to just this lake
        use_dataset = truth_data[truth_data["OBJECTID"] == float(lake_objectid)]

        mean_doc = use_dataset["DOC_MG_L"].mean()
        print(f"{lake_name} has a mean DOC of {mean_doc}. {len(use_dataset)} points.")
        mean_DOC_results.append(
            {
                "name": lake_name.upper(),
                "permanent_id": int(lake_info["PERMANENT_ID"]),
                "doc": mean_doc,
            }
        )

    pprint(mean_DOC_results)
