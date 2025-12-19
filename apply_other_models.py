# Steps to apply other models; Based heavily on https://github.com/leomet07/ny-satellite-chla-model

from tqdm import tqdm
import rasterio
import numpy as np
import pandas as pd
import os
from matplotlib import pyplot as plt
from shapely.geometry import Point
import rasterio.features
import sys
import joblib
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from scipy import stats
from matplotlib.ticker import ScalarFormatter

NAN_SUBSTITUTE_CONSANT = -99999
VISUALIZE = False

if len(sys.argv) <= 1:
    print(
        "You need to specify which algorithm out folders you are assembling as arguments."
    )
    sys.exit(1)

import inspect_shapefile

other_model = joblib.load(os.path.join("other_models", "ny-hab-cpu-model.joblib"))
lagos_lookup_table = pd.read_csv(
    os.path.join("other_models", "generated_lagoslookuptable.csv"),
    index_col="lagoslakei",  # for faster indexing
)

y_pred = []
y_true = []


def get_bands_from_tif(tif_path):
    with rasterio.open(tif_path) as src:
        profile = src.profile  # Get the profile of the existing raster
        transform = src.transform
        tags = src.tags()
        scale = tags["scale"]
        x_res = src.res[0]  # same as src.res[1]
        closest_insitu_date = tags["closest_insitu_date"]
        objectid = tags["objectid"]

        bands = src.read()

        # replace all -infs with nan
        bands[~np.isfinite(bands)] = np.nan

        if (
            "L2" in tif_path
        ):  # level 2 data correction https://www.usgs.gov/landsat-missions/landsat-collection-2-surface-reflectance
            bands = bands * 0.0000275 - 0.2  # operations on nan/inf are still nan/inf

        bands[bands > 0.1] = (
            np.nan
        )  # keep array shape but remove high reflectance outliers (clouds)
        # also this removes acolite outofbound nan values (10^36)

        return (
            bands,
            profile,
            transform,
            scale,
            x_res,
            closest_insitu_date,
            objectid,
        )


def get_constants_by_lagoslakeid(lakeid):
    lagos_lookup_table_filtered = lagos_lookup_table.loc[lakeid]

    matched_training_df = inspect_shapefile.truth_data[
        inspect_shapefile.truth_data["lagoslakeid"] == lakeid
    ]
    assert len(matched_training_df) > 0
    SA_SQ_KM = matched_training_df["AreaSqKm"].iloc[0]
    pct_dev = lagos_lookup_table_filtered["pct_dev"]
    pct_ag = lagos_lookup_table_filtered["pct_ag"]

    return SA_SQ_KM, pct_dev, pct_ag


def add_predictions_from_algorithim_out_folder(out_folder):
    algorithim_name = out_folder.split("_")[-1].upper().replace("/", "")
    subfolders = list(os.listdir(out_folder))
    subfolders.sort()
    all_files = []
    for subfolder in subfolders:
        if os.path.isfile(os.path.join(out_folder, subfolder)):
            continue  # this is the log file
        if subfolder == "rondaxe,_lake_tifs" or subfolder == "otter_lake_tifs":
            continue  # temporary, rondaxe does not have enough pixels around centroid

        tif_folder_path = os.path.join(out_folder, subfolder)

        for filename in os.listdir(tif_folder_path):
            tif_filepath = os.path.join(tif_folder_path, filename)
            all_files.append(tif_filepath)

    for tif_filepath in tqdm(all_files):
        try:
            (
                bands,
                profile,
                transform,
                scale,
                x_res,
                closest_insitu_date,
                objectid,
            ) = get_bands_from_tif(tif_filepath)
        except rasterio.errors.RasterioIOError:
            continue

        # match lagoslakeid
        lagoslakeid = inspect_shapefile.truth_data[
            (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
        ]["lagoslakeid"].iloc[0]

        # matched doc
        all_chla = inspect_shapefile.truth_data[
            (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
            & (inspect_shapefile.truth_data["DATE_SMP"] == closest_insitu_date)
        ]["CHL_A_UG_L"]
        try:
            true_chla = all_chla.item()
        except ValueError:  # array either has 2+ or 0 items
            if len(all_chla) > 0:  # means 2+ measurements for that date, take mean
                true_chla = all_chla.mean()
            else:
                raise Exception("No CHLA values found for that date.")
        # get lat and long
        centroid_lat = inspect_shapefile.truth_data[
            (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
            & (inspect_shapefile.truth_data["DATE_SMP"] == closest_insitu_date)
        ]["Lat-Cent"].iloc[
            0
        ]  # take first entry, lake centroid lat will be the same for any matched insitu
        centroid_long = inspect_shapefile.truth_data[
            (inspect_shapefile.truth_data["OBJECTID"] == float(objectid))
            & (inspect_shapefile.truth_data["DATE_SMP"] == closest_insitu_date)
        ]["Lon-Cent"].iloc[
            0
        ]  # take first entry, lake centroid long will be the same for any matched insitu

        radius_in_meters = 60
        circle = Point(centroid_long, centroid_lat).buffer(
            x_res * (radius_in_meters / float(scale))
        )  # however many x_res sized pixels needed for buffer of radius at downloaded scale

        outside_circle_mask = rasterio.features.geometry_mask(
            [circle], bands[0].shape, transform
        )

        for band in bands:
            band[outside_circle_mask] = (
                np.nan
            )  # arrays store pointer to ratio array, this is okay bc just a mutation

        all_valid_pixels_mask = np.isfinite(
            bands[0]
        )  # 2d mask to exclude clouds over centroid and outside centroid

        not_enough_pixels = False
        for band in bands:
            valid_pixels = band[np.isfinite(band)]
            if len(valid_pixels) < 3:
                not_enough_pixels = True

        if not_enough_pixels:
            continue

        sentinel_and_constant_bands = np.full(
            shape=(
                7,
                bands.shape[1],
                bands.shape[2],
            ),  # add 7 bands to make this 12 bands total
            fill_value=NAN_SUBSTITUTE_CONSANT,
            dtype=np.float64,
        )
        modified_bands = np.vstack([bands, sentinel_and_constant_bands])

        SA_SQ_KM_constant, pct_dev_constant, pct_ag_constant = (
            get_constants_by_lagoslakeid(int(lagoslakeid))
        )
        SA_SQ_KM_constant = 3  # manual override

        modified_bands[9] = np.full_like(
            modified_bands[0], SA_SQ_KM_constant, dtype=modified_bands.dtype
        )
        modified_bands[10] = np.full_like(
            modified_bands[0], pct_dev_constant, dtype=modified_bands.dtype
        )
        modified_bands[11] = np.full_like(
            modified_bands[0], pct_ag_constant, dtype=modified_bands.dtype
        )

        n_bands, n_rows, n_cols = modified_bands.shape
        n_samples = n_rows * n_cols
        raster_data_2d = modified_bands.transpose(1, 2, 0).reshape((n_samples, n_bands))
        raster_data_2d[~np.isfinite(raster_data_2d)] = (
            NAN_SUBSTITUTE_CONSANT  # make sure model runs even for bad pixels
        )
        # perform the prediction
        predictions = other_model.predict(raster_data_2d)
        predictions_raster = predictions.reshape(n_rows, n_cols)

        predictions_raster[~all_valid_pixels_mask] = np.nan

        if VISUALIZE:
            min_cbar_value = 0
            max_cbar_value = 60
            plt.imshow(
                predictions_raster,
                cmap="viridis",
                vmin=min_cbar_value,
                vmax=max_cbar_value,
            )
            plt.colorbar()
            plt.title(f"Predicted values for lake{objectid} on {closest_insitu_date}")
            plt.show()

        # get mean chla
        mean_chla = np.nanmean(predictions_raster)

        y_pred.append(mean_chla)
        y_true.append(true_chla)


for folder in sys.argv[1:]:
    add_predictions_from_algorithim_out_folder(folder)

r2 = r2_score(y_true, y_pred)
rmse = mean_squared_error(y_true, y_pred) ** 0.5
mae = mean_absolute_error(y_true, y_pred)

print("Number of predictions: ", len(y_pred))
print(f"r2 score: {r2}")
print(f"RMSE: {rmse}")
print(f"MAE: {mae}")

# scattergram
plt.figure(
    "Comparing Model's Prediction on ALTM Data to Corresponding Measured Value", (14, 7)
)
values = np.vstack([y_true, y_pred])
kernel = stats.gaussian_kde(values, bw_method=0.02)(values)

plt.scatter(y_true, y_pred, s=20, c=kernel, cmap="viridis")
plt.axline((0, 0), (50, 50), linewidth=2, color="red")
colorbar = plt.colorbar()
colorbar.set_label("Density", rotation=270, labelpad=15, fontweight="bold")

plt.xlim(0.1, 8)  # starts at 0.1 bc this is LOG scale and 0 is invalid
plt.ylim(0.1, 8)  # starts at 0.1 bc this is LOG scale and 0 is invalid
plt.xlabel("Observed Chl-a (µg/L)", fontweight="bold")
plt.ylabel("Predicted Chl-a (µg/L)", fontweight="bold")
plt.gca().set_aspect("equal")
plt.tight_layout()
plt.show()
