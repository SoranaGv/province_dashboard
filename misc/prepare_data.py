import geopandas as gpd
import json
import logging
from pathlib import Path


# -------------------------------------------------------------------
# Logging setup
# -------------------------------------------------------------------
def setup_logging(level=logging.INFO):
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )


# -------------------------------------------------------------------
# Data loading
# -------------------------------------------------------------------
def load_data(path: str) -> gpd.GeoDataFrame:
    logging.info(f"Loading data from {path}")
    gdf = gpd.read_file(path)
    logging.info(f"Loaded {len(gdf)} records")
    return gdf


# -------------------------------------------------------------------
# CRS handling
# -------------------------------------------------------------------
def ensure_wgs84(gdf: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    if gdf.crs is None:
        raise ValueError("GeoDataFrame has no CRS defined")

    epsg = gdf.crs.to_epsg()
    if epsg != 4326:
        logging.info(f"Converting CRS from EPSG:{epsg} to EPSG:4326")
        gdf = gdf.to_crs(epsg=4326)
    else:
        logging.info("CRS already EPSG:4326")

    return gdf


# -------------------------------------------------------------------
# GeoJSON creation
# -------------------------------------------------------------------
def create_geojson(gdf: gpd.GeoDataFrame) -> dict:
    logging.info("Converting GeoDataFrame to GeoJSON")
    return json.loads(gdf.to_json())


# -------------------------------------------------------------------
# Save output
# -------------------------------------------------------------------
def save_geojson(geojson: dict, output_path: str):
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    logging.info(f"Saving GeoJSON to {output_path}")
    with open(output_path, "w") as f:
        json.dump(geojson, f)


# -------------------------------------------------------------------
# Entry point / coordinator
# -------------------------------------------------------------------
def main(
    input_path: str = "../data/parcels_scenarios_clean.gpkg",
    output_path: str = "../data/parcels.geojson",
):
    setup_logging()

    logging.info("Starting parcel processing pipeline")

    gdf = load_data(input_path)
    gdf = ensure_wgs84(gdf)
    geojson = create_geojson(gdf)
    save_geojson(geojson, output_path)

    logging.info("Pipeline completed successfully")
    logging.info(f"Columns: {gdf.columns.tolist()}")


# -------------------------------------------------------------------
# Script execution
# -------------------------------------------------------------------
if __name__ == "__main__":
    main()