import json
import logging

import dash
import geopandas as gpd
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from dash import Input, Output, State, callback, dcc, html

from misc.prepare_data import create_geojson, ensure_wgs84, load_data, setup_logging


# -------------------------------------------------------------------
# Map creation
# -------------------------------------------------------------------
def create_map(
    gdf,
    geojson,
    parcel_modifications,
    effective_co2,
    wmu_geojson,
    selected_ids=None,
    highlight_ids=None,
):
    selected_ids = selected_ids or []
    highlight_ids = highlight_ids or []

    # Pre-calculate centroids
    if "centroid_lat" not in gdf.columns:
        centroids = gdf.geometry.centroid
        gdf["centroid_lat"] = centroids.y
        gdf["centroid_lon"] = centroids.x

    # Initialize empty figure
    fig = go.Figure()

    # Main parcel layer
    fig.add_trace(
        go.Choroplethmapbox(
            geojson=geojson,
            locations=gdf["AAN_ID"],
            z=effective_co2,
            featureidkey="properties.AAN_ID",
            colorscale="Viridis",
            zmin=effective_co2.min(),
            zmax=effective_co2.max(),
            marker_opacity=0.85,
            marker_line_width=0.5,
            marker_line_color="white",
            colorbar=dict(
                title="CO2 Emissions<br>(tons)", x=1.02, thickness=25, len=0.8
            ),
            text=gdf["AAN_ID"],
            customdata=list(
                zip(
                    gdf["SUM_CO2_totaal"],
                    effective_co2,
                    gdf["FIRST_zomerdrooglegging"],
                    gdf["FIRST_winterdrooglegging"],
                    gdf["FIRST_Oppervlakte__m2_"],
                    gdf["FIRST_Dekkingsgraad_organisch__"],
                    gdf["category"],
                )
            ),
            hovertemplate=(
                "<b>Parcel %{text}</b><br><br>"
                "Area: %{customdata[4]:.2f} ha<br>"
                "Soil Type: %{customdata[6]:text}<br>"
                "Coverage: %{customdata[5]:.2f}<br><br>"
                "Baseline summer drainage: %{customdata[2]:.1f} m<br>"
                "Baseline winter drainage: %{customdata[3]:.1f} m<br><br>"
                "Baseline CO₂: %{customdata[0]:,.0f} t<br>"
                "Current CO₂: %{customdata[1]:,.0f} t<br>"
                "<extra></extra>"
            ),
            selectedpoints=selected_ids if selected_ids else None,
        )
    )

    # Compute map center from total bounds of the GeoDataFrame
    bounds = gdf.total_bounds
    center_lat = (bounds[1] + bounds[3]) / 2
    center_lon = (bounds[0] + bounds[2]) / 2

    # Water Management Units layer
    fig.update_layout(
        mapbox_layers=[
            {
                "source": wmu_geojson,
                "type": "line",
                "color": "rgba(0,0,0,0)",
                "line": {"width": 2},
            }
        ]
    )

    # UI buttons for toggling the Unit layer on and off
    fig.update_layout(
        updatemenus=[
            dict(
                type="buttons",
                direction="right",
                x=0.01,
                y=0.99,
                xanchor="left",
                yanchor="top",
                pad=dict(r=6, t=6),
                buttons=[
                    dict(
                        label="Show Management Units",
                        method="relayout",
                        args=[{"mapbox.layers[0].color": "#313033"}],
                    ),
                    dict(
                        label="Hide Management Units",
                        method="relayout",
                        args=[{"mapbox.layers[0].color": "rgba(0,0,0,0)"}],
                    ),
                ],
            )
        ]
    )

    # Base map layout and interaction settings
    fig.update_layout(
        mapbox=dict(
            style="carto-positron", center=dict(lat=center_lat, lon=center_lon), zoom=10
        ),
        margin=dict(l=0, r=0, t=0, b=0),
        height=600,
        clickmode="event+select",
        dragmode="pan",
        selectdirection="any",
        showlegend=False,
        uirevision="constant",
        transition={"duration": 300},
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
    )

    return fig


# -------------------------------------------------------------------
# Effective CO2 calculation
# -------------------------------------------------------------------
def compute_effective_co2(gdf, parcel_modifications, scenarios_indexed):
    effective_co2 = gdf["SUM_CO2_totaal"].copy()

    for idx, mod in parcel_modifications.items():
        summer = mod["summer"]
        infiltration = mod["infiltration"]

        if summer == 0:
            continue

        row = gdf.iloc[idx]
        aan_id = row["AAN_ID"]
        baseline_summer = row["FIRST_zomerdrooglegging"]
        baseline_winter = row["FIRST_winterdrooglegging"]

        # Calculate target summer drainage
        target_summer = round(
            baseline_summer - summer, 6
        )  # Round to avoid floating-point issues

        try:
            # OPTIMIZED: O(1) index lookup instead of O(n) filtering
            co2_value = scenarios_indexed.loc[
                (aan_id, target_summer, baseline_winter, infiltration), "CO2_parcel"
            ]
            effective_co2.iloc[idx] = co2_value
        except KeyError:
            # No matching scenario found - keep baseline
            logging.debug(
                f"No scenario match for parcel {aan_id}: "
                f"summer={target_summer}, winter={baseline_winter}, infiltration={infiltration}"
            )

    return effective_co2


# -------------------------------------------------------------------
# App factory
# -------------------------------------------------------------------
def create_app(gdf, geojson, scenarios_df, wmu_geojson):
    aan_id_to_idx = {aan_id: idx for idx, aan_id in enumerate(gdf["AAN_ID"])}

    wmu_to_parcels = gdf.groupby("Code_1").apply(lambda x: x.index.tolist()).to_dict()

    idx_to_wmu = gdf["Code_1"].to_dict()

    scenarios_df["zomerdrooglegging"] = scenarios_df["zomerdrooglegging"].round(6)
    scenarios_df["winterdrooglegging"] = scenarios_df["winterdrooglegging"].round(6)

    scenarios_indexed = scenarios_df.set_index(
        ["AAN_ID", "zomerdrooglegging", "winterdrooglegging", "infiltratiemaatregel"]
    ).sort_index()

    app = dash.Dash(__name__, external_stylesheets=["assets/styles.css"])
    app.title = "Groningen Peat CO2 Dashboard"

    app.layout = html.Div(
        [
            # Header section
            html.Div(
                [
                    html.H1(
                        "Groningen Peat CO2 Emissions Dashboard",
                        className="header-title",
                    ),
                    html.P(
                        "Select parcels, choose the water level and infiltration type, then click 'Apply Modification'",
                        className="header-subtitle",
                    ),
                ],
                className="header-container",
            ),
            # Total Emissions Summary
            html.Div(
                [
                    html.Div(
                        [
                            html.H4("Total Current CO2"),
                            html.H2(
                                id="total-current-co2", className="emissions-current"
                            ),
                        ],
                        className="emissions-box",
                    ),
                    html.Div(
                        [
                            html.H4("Total Effective CO2"),
                            html.H2(
                                id="total-effective-co2",
                                className="emissions-effective",
                            ),
                        ],
                        className="emissions-box",
                    ),
                    html.Div(
                        [
                            html.H4("Total Reduction"),
                            html.H2(
                                id="total-reduction", className="emissions-reduction"
                            ),
                        ],
                        className="emissions-box",
                    ),
                ],
                className="emissions-summary",
            ),
            # Main content area
            html.Div(
                [
                    # Left panel - Map
                    html.Div(
                        [
                            dcc.Loading(
                                id="loading-map",
                                type="default",
                                children=[
                                    dcc.Graph(
                                        id="map",
                                        className="map-container",
                                        config={
                                            "scrollZoom": True,
                                            "modeBarButtonsToAdd": [
                                                "select2d",
                                                "lasso2d",
                                            ],
                                            "displayModeBar": True,
                                            "displaylogo": False,
                                        },
                                    ),
                                ],
                            ),
                            html.Button(
                                "Clear Selection",
                                id="clear-btn",
                                n_clicks=0,
                                className="clear-button",
                            ),
                        ],
                        className="map-panel",
                    ),
                    # Right side - Controls
                    html.Div(
                        [
                            html.Label("Raise water level by:"),
                            dcc.Dropdown(
                                id="summer-selector",
                                clearable=False,
                                placeholder="Select delta",
                            ),
                            html.Label("Select infiltration type:"),
                            dcc.Dropdown(
                                id="infiltration-select",
                                options=[
                                    {
                                        "label": "Reference (no infiltration)",
                                        "value": "ref",
                                    },
                                    {"label": "Active infiltration", "value": "AWIS"},
                                    {"label": "Passive infiltration", "value": "PWIS"},
                                ],
                                value="ref",
                                clearable=False,
                            ),
                            html.Div(
                                id="button-container",
                                children=[
                                    html.Button(
                                        "Apply Modification to Selected",
                                        id="apply-btn",
                                        n_clicks=0,
                                        className="apply-button",
                                    ),
                                    html.Button(
                                        "Reset All Modifications",
                                        id="reset-modifications-btn",
                                        n_clicks=0,
                                        className="reset-button",
                                    ),
                                ],
                            ),
                            # Selection statistics
                            html.Div(
                                [
                                    html.H3("Current Selection"),
                                    html.Div(
                                        [
                                            html.H4("Selected Parcels"),
                                            html.H2(
                                                id="selected-count",
                                                children="0",
                                            ),
                                        ],
                                        className="stat-item",
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Selection CO2 (Current)"),
                                            html.Div(
                                                id="selection-co2-current",
                                                className="stat-value",
                                            ),
                                        ],
                                        className="stat-item",
                                    ),
                                    html.Div(
                                        [
                                            html.H4("Selection CO2 (if Applied)"),
                                            html.Div(
                                                id="selection-co2-scenario",
                                                className="stat-value",
                                            ),
                                        ],
                                        className="stat-item",
                                    ),
                                ],
                                className="selection-stats",
                            ),
                            # Modified parcels list
                            html.Div(
                                [
                                    html.H3("Modified Parcels"),
                                    html.Div(
                                        id="modified-list",
                                        className="modified-list",
                                    ),
                                ],
                                className="modified-panel",
                            ),
                        ],
                        className="controls-panel",
                    ),
                ],
                className="container",
            ),
            # Hidden stores
            dcc.Store(id="selected-parcels", data=[]),
            dcc.Store(id="parcel-modifications", data={}),
            dcc.Store(id="highlight-parcels", data=[]),
        ]
    )

    # -------------------------------------------------------------------
    # Callback 1: OPTIMIZED Selection handling
    # -------------------------------------------------------------------
    @app.callback(
        Output("selected-parcels", "data"),
        Output("highlight-parcels", "data"),
        Output("map", "selectedData"),
        Output("map", "clickData"),
        [
            Input("map", "clickData"),
            Input("map", "selectedData"),
            Input("clear-btn", "n_clicks"),
        ],
        State("selected-parcels", "data"),
        State("highlight-parcels", "data"),
        prevent_initial_call=True,
    )
    def handle_selection(
        clickData, selectedData, clear_clicks, selected_ids, highlight_ids
    ):
        ctx = dash.callback_context

        if selected_ids is None:
            selected_ids = []

        if highlight_ids is None:
            highlight_ids = []

        if ctx.triggered:
            trigger = ctx.triggered[0]["prop_id"]

            if trigger == "clear-btn.n_clicks":
                return [], [], None, None

            elif trigger == "map.clickData" and clickData is not None:
                aan_id = clickData["points"][0]["location"]

                # OPTIMIZED: O(1) lookup instead of O(n) search
                idx = aan_id_to_idx.get(aan_id)
                if idx is None:
                    return selected_ids, highlight_ids, None, None

                # OPTIMIZED: O(1) lookup for WMU code
                wm_unit = idx_to_wmu[idx]

                # OPTIMIZED: O(1) lookup for all parcels in unit
                same_unit_indices = wmu_to_parcels[wm_unit]

                # Toggle logic
                if set(same_unit_indices).issubset(set(selected_ids)):
                    selected_ids = [
                        i for i in selected_ids if i not in same_unit_indices
                    ]
                    highlight_ids = []
                else:
                    selected_ids = list(set(selected_ids).union(same_unit_indices))
                    highlight_ids = [i for i in same_unit_indices if i != idx]

            elif trigger == "map.selectedData" and selectedData is not None:
                selected_aan_ids = [p["location"] for p in selectedData["points"]]

                selected_idx_set = set()
                highlight_idx_set = set()

                for aan_id in selected_aan_ids:
                    # OPTIMIZED: O(1) lookups
                    idx = aan_id_to_idx.get(aan_id)
                    if idx is None:
                        continue

                    wm_unit = idx_to_wmu[idx]
                    same_unit_indices = wmu_to_parcels[wm_unit]

                    selected_idx_set.update(same_unit_indices)
                    highlight_idx_set.update(i for i in same_unit_indices if i != idx)

                selected_ids = list(selected_idx_set)
                highlight_ids = list(highlight_idx_set)

        return selected_ids, highlight_ids, None, None

    # -------------------------------------------------------------------
    # Callback 2: Modifications handling
    # -------------------------------------------------------------------
    @app.callback(
        [
            Output("parcel-modifications", "data"),
            Output("selected-parcels", "data", allow_duplicate=True),
            Output("highlight-parcels", "data", allow_duplicate=True),
        ],
        [
            Input("apply-btn", "n_clicks"),
            Input("reset-modifications-btn", "n_clicks"),
        ],
        [
            State("selected-parcels", "data"),
            State("parcel-modifications", "data"),
            State("summer-selector", "value"),
            State("infiltration-select", "value"),
        ],
        prevent_initial_call=True,
    )
    def handle_modifications(
        apply_clicks,
        reset_clicks,
        selected_ids,
        parcel_mods,
        summer_val,
        infiltration_val,
    ):
        ctx = dash.callback_context

        if parcel_mods is None:
            parcel_mods = {}
        if selected_ids is None:
            selected_ids = []

        parcel_mods = {int(k): v for k, v in parcel_mods.items()}

        highlight_ids = []

        if ctx.triggered:
            trigger = ctx.triggered[0]["prop_id"]

            if trigger == "apply-btn.n_clicks" and selected_ids:
                for idx in selected_ids:
                    parcel_mods[idx] = {
                        "summer": summer_val,
                        "infiltration": infiltration_val,
                    }
                selected_ids = []
                highlight_ids = []

            elif trigger == "reset-modifications-btn.n_clicks":
                parcel_mods = {}
                selected_ids = []
                highlight_ids = []

        return parcel_mods, selected_ids, highlight_ids

    # -------------------------------------------------------------------
    # Callback 3: Summer selector update
    # -------------------------------------------------------------------
    @app.callback(
        Output("summer-selector", "options"),
        Output("summer-selector", "value"),
        Input("selected-parcels", "data"),
    )
    def update_summer_selector(selected_ids):
        if not selected_ids:
            return [{"label": "0.0 m", "value": 0.0}], 0.0

        min_summer = gdf.iloc[selected_ids]["FIRST_zomerdrooglegging"].dropna().min()

        max_delta = max(min_summer - 0.2, 0)

        if max_delta <= 0:
            return [{"label": "0.0 m", "value": 0.0}], 0.0

        values = np.round(np.arange(0.0, max_delta + 0.01, 0.1), 1)

        options = [{"label": f"{v:.1f} m", "value": float(v)} for v in values]

        return options, options[0]["value"]

    # -------------------------------------------------------------------
    # Callback 4: Update display (map + statistics)
    # -------------------------------------------------------------------
    @app.callback(
        [
            Output("map", "figure"),
            Output("selected-count", "children"),
            Output("selection-co2-current", "children"),
            Output("selection-co2-scenario", "children"),
            Output("modified-list", "children"),
            Output("total-current-co2", "children"),
            Output("total-effective-co2", "children"),
            Output("total-reduction", "children"),
        ],
        [
            Input("selected-parcels", "data"),
            Input("highlight-parcels", "data"),
            Input("parcel-modifications", "data"),
            Input("summer-selector", "value"),
            Input("infiltration-select", "value"),
        ],
    )
    def update_display(
        selected_ids, highlight_ids, parcel_mods, summer_val, infiltration_val
    ):
        if selected_ids is None:
            selected_ids = []
        if highlight_ids is None:
            highlight_ids = []
        if parcel_mods is None:
            parcel_mods = {}

        parcel_mods = {int(k): v for k, v in parcel_mods.items()}

        effective_co2 = compute_effective_co2(gdf, parcel_mods, scenarios_indexed)

        fig = create_map(
            gdf,
            geojson,
            parcel_mods,
            effective_co2,
            wmu_geojson,
            selected_ids,
            highlight_ids,
        )

        total_current = gdf["SUM_CO2_totaal"].sum()
        total_effective = effective_co2.sum()
        total_reduction_amount = total_current - total_effective
        total_reduction_pct = (
            (total_reduction_amount / total_current * 100) if total_current > 0 else 0
        )

        if selected_ids:
            sel = gdf.iloc[selected_ids]
            count = len(sel)
            selection_current = sel["SUM_CO2_totaal"].sum()

            selection_scenario = compute_effective_co2(
                sel,
                {
                    i: {"summer": summer_val, "infiltration": infiltration_val}
                    for i in range(len(sel))
                },
                scenarios_indexed,
            ).sum()

            selection_current_text = f"{selection_current:,.0f} tons"
            if selection_current > 0:
                reduction_pct = (
                    (selection_current - selection_scenario) / selection_current * 100
                )
                selection_scenario_text = (
                    f"{selection_scenario:,.0f} tons ({reduction_pct:.1f}% reduction)"
                )
            else:
                selection_scenario_text = f"{selection_scenario:,.0f} tons"

        else:
            count = 0
            selection_current_text = "No parcels selected"
            selection_scenario_text = "No parcels selected"

        if parcel_mods:
            modified_list = [
                html.Div(
                    [
                        html.Span(
                            f"Parcel {gdf.iloc[idx]['AAN_ID']}: ",
                            className="modified-item-title",
                        ),
                        html.Span("Modified", className="modified-item-status"),
                        html.Br(),
                        html.Span(
                            f"Was: {gdf.iloc[idx]['SUM_CO2_totaal']:.0f} tons -> ",
                            className="modified-item-before",
                        ),
                        html.Span(
                            f"Now: {effective_co2.iloc[idx]:.0f} tons",
                            className="modified-item-after",
                        ),
                    ],
                    className="modified-item",
                )
                for idx in parcel_mods.keys()
            ]
        else:
            modified_list = [
                html.P(
                    "No modifications applied yet",
                    className="no-modifications",
                )
            ]

        return (
            fig,
            str(count),
            selection_current_text,
            selection_scenario_text,
            modified_list,
            f"{total_current:,.0f} tons",
            f"{total_effective:,.0f} tons",
            f"{total_reduction_amount:,.0f} tons ({total_reduction_pct:.1f}%)",
        )

    return app


# -------------------------------------------------------------------
# Entry point
# -------------------------------------------------------------------
def main():
    setup_logging()
    logging.info("Starting Dash application")

    gdf = gpd.read_file("data/dashboard_Data.gpkg", layer="main")
    gdf = ensure_wgs84(gdf)
    geojson = create_geojson(gdf)

    gdf["SUM_CO2_totaal"] = (
        pd.to_numeric(gdf["SUM_CO2_totaal"], errors="coerce") / 1000.0
    )
    gdf["FIRST_Dekkingsgraad_organisch__"] = (
        pd.to_numeric(gdf["FIRST_Dekkingsgraad_organisch__"], errors="coerce") / 100
    )
    gdf["FIRST_Oppervlakte__m2_"] = (
        pd.to_numeric(gdf["FIRST_Oppervlakte__m2_"], errors="coerce") / 1000.0
    )

    wmu_gdf = gpd.read_file("data/dashboard_Data.gpkg", layer="waterManagementUnits")
    wmu_gdf = ensure_wgs84(wmu_gdf)
    wmu_geojson = json.loads(wmu_gdf.to_json())

    scenarios_df = pd.read_csv("data/parcel_co2_scenarios.csv", sep=";", decimal=",")

    scenarios_df["CO2_parcel"] = (
        pd.to_numeric(scenarios_df["CO2_parcel"], errors="coerce") / 1000
    )
    scenarios_df["zomerdrooglegging"] = pd.to_numeric(
        scenarios_df["zomerdrooglegging"], errors="coerce"
    )
    scenarios_df["winterdrooglegging"] = pd.to_numeric(
        scenarios_df["winterdrooglegging"], errors="coerce"
    )

    app = create_app(gdf, geojson, scenarios_df, wmu_geojson)
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
