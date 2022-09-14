""" Core module for the dash application """
import dash
import dash_bootstrap_components as dbc

dashapp = dash.Dash(
    __name__,
    external_stylesheets=[dbc.themes.BOOTSTRAP],
    suppress_callback_exceptions=True,
)
