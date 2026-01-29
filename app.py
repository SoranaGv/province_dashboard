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
            # Hover information for a parcel via customdata
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
             # Force empty list when nothing selected
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
# Effective CO2 calculation based on summer drainage, winter drainage and infiltration parameters
# -------------------------------------------------------------------

def compute_effective_co2(gdf, parcel_modifications, scenarios_indexed):
    effective_co2 = gdf["SUM_CO2_totaal"].copy()

    # Iterate over parcels and extract modification parameters for this parcel
    for idx, mod in parcel_modifications.items():
        summer = mod["summer"]
        infiltration = mod["infiltration"]

        if summer == 0:
            continue
        
        # Select the current parcel by its row index and retrieve its attributes
        row = gdf.iloc[idx]
        aan_id = row["AAN_ID"]
        baseline_summer = row["FIRST_zomerdrooglegging"]
        baseline_winter = row["FIRST_winterdrooglegging"]

        # Calculate target summer drainage
        target_summer = round(
            baseline_summer - summer, 6
        )  # Round to avoid floating-point issues

        try:
            # Lookup precomputed CO2 for parcel & scenario
            co2_value = scenarios_indexed.loc[
                (aan_id, target_summer, baseline_winter, infiltration), "CO2_parcel"
            ]
            effective_co2.iloc[idx] = co2_value
        except KeyError:
            # No matching scenario: retain baseline CO2
            logging.debug(
                f"No scenario match for parcel {aan_id}: "
                f"summer={target_summer}, winter={baseline_winter}, infiltration={infiltration}"
            )

    return effective_co2


# -------------------------------------------------------------------
# App factory
# -------------------------------------------------------------------

def create_app(gdf, geojson, scenarios_df, wmu_geojson):
    # Map AAN_ID to row index for quick lookup
    aan_id_to_idx = {aan_id: idx for idx, aan_id in enumerate(gdf["AAN_ID"])}

    # Group parcel indices by water management unit
    wmu_to_parcels = gdf.groupby("Code_1").apply(lambda x: x.index.tolist()).to_dict()

    # Map each row index to its water management unit
    idx_to_wmu = gdf["Code_1"].to_dict()

    scenarios_df["zomerdrooglegging"] = scenarios_df["zomerdrooglegging"].round(6)
    scenarios_df["winterdrooglegging"] = scenarios_df["winterdrooglegging"].round(6)

    # Create multi-level index for fast scenario lookup by parcel and parameters
    scenarios_indexed = scenarios_df.set_index(
        ["AAN_ID", "zomerdrooglegging", "winterdrooglegging", "infiltratiemaatregel"]
    ).sort_index()

     # Initialize Dash application
    app = dash.Dash(__name__, external_stylesheets=["assets/styles.css"])
    app.title = "Groningen Peat CO2 Dashboard"

    app.layout = html.Div(
        [
            # Header section: title and short instructions
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
            # Total Emissions Summary, displays aggregated current, effective, and reduced CO2 values
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
            # Main content area: map (left) and controls (right)
            html.Div(
                [
                    # Left panel - interactive Map
                    html.Div(
                        [
                            dcc.Loading(
                                id="loading-map",
                                type="default",
                                children=[
                                    dcc.Graph(
                                        id="map",
                                        className="map-container",
                                        # Add selection tools to the mode bar
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
                    # Right side - User Controls
                    html.Div(
                        [   
                            # Dropdown for selecting water level increase
                            html.Label("Raise water level by:"),
                            dcc.Dropdown(
                                id="summer-selector",
                                clearable=False,
                                placeholder="Select delta",
                            ),
                            # Dropdown for selecting infiltration measure
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
                            # Action buttons: apply or reset parcel modifications
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
                            # Selection statistics panel, shows stats for currently selected parcels
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
                            # List of parcels that have modifications applied
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
            # Hidden client-side stores for application state
            dcc.Store(id="selected-parcels", data=[]),
            dcc.Store(id="parcel-modifications", data={}),
            dcc.Store(id="highlight-parcels", data=[]),
        ]
    )


    # -------------------------------------------------------------------
    # Callback 1: 
    # Handles all parcel selection logic (Single-click selections, Box / lasso selection, Clear selection button)
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

        # Only react when something actually triggered the callback
        if ctx.triggered:
            trigger = ctx.triggered[0]["prop_id"]

            # Clear button: reset all selection and highlights
            if trigger == "clear-btn.n_clicks":
                return [], [], None, None

            # Single click on a parcel, it expands selection to the entire management unit
            elif trigger == "map.clickData" and clickData is not None:
                aan_id = clickData["points"][0]["location"]

                # Optimized lookup, now O(1) for all tables
                idx = aan_id_to_idx.get(aan_id)
                if idx is None:
                    return selected_ids, highlight_ids, None, None
                wm_unit = idx_to_wmu[idx]
                same_unit_indices = wmu_to_parcels[wm_unit]

                # Toggle logic:
                # If entire unit is already selected, deselect it
                # Otherwise, add entire unit to selection
                if set(same_unit_indices).issubset(set(selected_ids)):
                    selected_ids = [
                        i for i in selected_ids if i not in same_unit_indices
                    ]
                    highlight_ids = []
                else:
                    selected_ids = list(set(selected_ids).union(same_unit_indices))
                    highlight_ids = [i for i in same_unit_indices if i != idx]

            # Box / lasso selection, it expands selection to full management units for all parcels
            elif trigger == "map.selectedData" and selectedData is not None:
                selected_aan_ids = [p["location"] for p in selectedData["points"]]

                # Use sets to avoid duplicate indices
                selected_idx_set = set()
                highlight_idx_set = set()

                for aan_id in selected_aan_ids:
                    # Optimized lookups, now O(1)
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
    # Callback 2: 
    # Handles applying and resetting parcel modifications 
    # (Apply button assigns summer/infiltration settings to all selected parcels, Reset button removes all modifications)
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

        # JSON stores keys as strings → convert back to integers
        parcel_mods = {int(k): v for k, v in parcel_mods.items()}

        highlight_ids = []

        if ctx.triggered:
            trigger = ctx.triggered[0]["prop_id"]

            # Apply modifications to all currently selected parcels
            if trigger == "apply-btn.n_clicks" and selected_ids:
                for idx in selected_ids:
                    parcel_mods[idx] = {
                        "summer": summer_val,
                        "infiltration": infiltration_val,
                    }
                # Clear selection and highlights after applying
                selected_ids = []
                highlight_ids = []

            # Reset all parcel modifications
            elif trigger == "reset-modifications-btn.n_clicks":
                parcel_mods = {}
                selected_ids = []
                highlight_ids = []

        return parcel_mods, selected_ids, highlight_ids


    # -------------------------------------------------------------------
    # Callback 3: 
    # Updates the summer water-level selector based on the current selection.
    # ------------------------------------------------------------------- 

    @app.callback(
        Output("summer-selector", "options"),
        Output("summer-selector", "value"),
        Input("selected-parcels", "data"),
    )
    def update_summer_selector(selected_ids):
        if not selected_ids:
            return [{"label": "0.0 m", "value": 0.0}], 0.0

        # Determine the minimum baseline summer drainage across all selected parcels
        min_summer = gdf.iloc[selected_ids]["FIRST_zomerdrooglegging"].dropna().min()

        # Drainage cannot go below 0.2
        max_delta = max(min_summer - 0.2, 0)

        # If no reduction is possible, lock selector to 0.0 m
        if max_delta <= 0:
            return [{"label": "0.0 m", "value": 0.0}], 0.0
        
        # Generate allowed delta values: Start at 0.0, Step by 0.1 m, Round to avoid floating-point errors
        values = np.round(np.arange(0.0, max_delta + 0.01, 0.1), 1)

        # Convert numeric values into dropdown options
        options = [{"label": f"{v:.1f} m", "value": float(v)} for v in values]

        return options, options[0]["value"]


    # -------------------------------------------------------------------
    # Callback 4: 
    # Updates the entire dashboard display whenever selection, modifications, or scenario inputs change.
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

        # JSON stores keys as strings, convert back to integers
        parcel_mods = {int(k): v for k, v in parcel_mods.items()}

        # Compute effective CO2 for new parameters
        effective_co2 = compute_effective_co2(gdf, parcel_mods, scenarios_indexed)

        # Create map
        fig = create_map(
            gdf,
            geojson,
            parcel_mods,
            effective_co2,
            wmu_geojson,
            selected_ids,
            highlight_ids,
        )

        # Total
        total_current = gdf["SUM_CO2_totaal"].sum()
        total_effective = effective_co2.sum()
        total_reduction_amount = total_current - total_effective
        total_reduction_pct = (
            (total_reduction_amount / total_current * 100) if total_current > 0 else 0
        )

        # Selection stats
        if selected_ids:
            sel = gdf.iloc[selected_ids]
            count = len(sel)
            selection_current = sel["SUM_CO2_totaal"].sum()

            # Compute CO2 if selected water level + infiltration parameters are applied
            selection_scenario = compute_effective_co2(
                sel,
                {
                    i: {"summer": summer_val, "infiltration": infiltration_val}
                    for i in range(len(sel))
                },
                scenarios_indexed,
            ).sum()
        
            selection_current_text = f"{selection_current:,.0f} tons"
            # Include reduction percentage if current CO2 > 0
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
            # No parcels selected, display placeholder messages
            count = 0
            selection_current_text = "No parcels selected"
            selection_scenario_text = "No parcels selected"

        # Build modified parcel list for display: shows parcel ID, modification status, and before/after CO2
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
            # Placeholder message when no modifications have been applied
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

    # Read parcels dataset with baseline parameters
    gdf = gpd.read_file("data/dashboard_Data.gpkg", layer="main")
    gdf = ensure_wgs84(gdf)
    geojson = create_geojson(gdf)

    # Convert columns to numeric
    gdf["SUM_CO2_totaal"] = (
        pd.to_numeric(gdf["SUM_CO2_totaal"], errors="coerce") / 1000.0
    )
    gdf["FIRST_Dekkingsgraad_organisch__"] = (
        pd.to_numeric(gdf["FIRST_Dekkingsgraad_organisch__"], errors="coerce") / 100
    )
    gdf["FIRST_Oppervlakte__m2_"] = (
        pd.to_numeric(gdf["FIRST_Oppervlakte__m2_"], errors="coerce") / 1000.0
    )

     # Read Water Management Units shapefile
    wmu_gdf = gpd.read_file("data/dashboard_Data.gpkg", layer="waterManagementUnits")
    wmu_gdf = ensure_wgs84(wmu_gdf)
    wmu_geojson = json.loads(wmu_gdf.to_json())

    # Read CO2 emission scenario file
    scenarios_df = pd.read_csv("data/parcel_co2_scenarios.csv", sep=";", decimal=",")

    # Convert columns to numeric
    scenarios_df["CO2_parcel"] = (
        pd.to_numeric(scenarios_df["CO2_parcel"], errors="coerce") / 1000
    )
    scenarios_df["zomerdrooglegging"] = pd.to_numeric(
        scenarios_df["zomerdrooglegging"], errors="coerce"
    )
    scenarios_df["winterdrooglegging"] = pd.to_numeric(
        scenarios_df["winterdrooglegging"], errors="coerce"
    )

    # Start the application
    app = create_app(gdf, geojson, scenarios_df, wmu_geojson)
    app.run(debug=True, host="0.0.0.0", port=8050)


if __name__ == "__main__":
    main()
