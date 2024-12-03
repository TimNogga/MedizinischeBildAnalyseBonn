import io

import dash
import skimage.io
from dash import dcc, html, Input, Output, State
import dash_bootstrap_components as dbc
import plotly.graph_objects as go
import flask
import os
import numpy as np
from skimage.color import rgb2gray
from skimage.filters import gaussian
from skimage.segmentation import active_contour
import plotly.express as px
import regex as re
import base64
import io

server = flask.Flask('app')
server.secret_key = os.environ.get('secret_key', 'secret')
app = dash.Dash('app', server=server, external_stylesheets=[dbc.icons.BOOTSTRAP])


app.layout = html.Div([
    html.H1('Active Contour Model'),
    html.Div(
        [
            dcc.Upload(
                children=html.P(['Upload Image']),
                id='img-upload',
                className='img-upload',
                style={'width': '100%', 'border': '1px solid', 'borderRadius': '5px', 'textAlign': 'center'},
            ),
            dbc.Button(id='close',  style={'visibility': 'hidden'})
        ],
        id='upload_container', style={'display': 'flex', 'flexDirection': 'row', 'height': '60px', 'width': '20%',
                                      'justifyContent': 'space-around', 'border': '1px solid', 'alignItems': 'center',
                                      'borderRadius': '5px'}),
    html.Div([
    html.Div([
        html.Div([
            html.P('Alpha'),
            dcc.Slider(0, 1, value=0.01, id="alpha")
        ],
            style={'width': '20%'}
        ),
        html.Div([
            html.P('Beta'),
            dcc.Slider(0, 1, value=0.1, id="beta")
        ],
            style={'width': '20%'}
        ),
        html.Div([
            html.P('w_line'),
            dcc.Slider(-5, 5, value=0, id="w_line")
        ],
            style={'width': '20%'}
        ),
        html.Div([
            html.P('w_edge'),
            dcc.Slider(-5, 5, value=1, id="w_edge")
        ],
            style={'width': '20%'}
        ),
        html.Div([
            html.P('sigma'),
            dcc.Slider(0.1, 5, value=1, id="sigma")
        ],
            style={'width': '20%'}
        ),
        html.Div([
            html.P('Stützstellen'),
            dcc.Slider(50, 1000, value=400, id="points")
        ],
            style={'width': '20%'}
        ),
        html.Div([
            html.Div([
                html.P('Select contour'),
                html.Button('Continue', id='continue'),
                html.Button('Stop', id='stop'),
            ], style={'display': 'flex', 'flexDirection': 'row', 'justifyContent': 'space-between'},
                id='hidden-continue'
            ),
            dcc.Slider(1, 10, value=1, id="s-conture")
        ],
            style={'width': '20%', 'visibility': 'hidden'},
            id="hidden-slider"
        )

    ], style={'display': 'flex', 'flexDirection': 'row'}),
    html.Button('Start ACM', id='get-contures'),

    dcc.Graph(id='picture', config={'modeBarButtonsToAdd': [
        'drawopenpath',
        'eraseshape'
    ], 'modeBarButtonsToRemove': []}, style={'height':'600px'}),
        ],style={'width': '100%', 'visibility': 'hidden'}, id='hidden-div'),

    dcc.Store('init'),
    dcc.Store('boundary_condition'),
    dcc.Store('graphs', data=[]),
    dcc.Store('img-store'),
    dcc.Interval('my-interval', interval=500, disabled=True),
    dbc.Alert(
                children="",
                id="alert",
                is_open=False,
                duration=3000,
                style={'position':'fixed', 'bottom': '10px', 'left':'10px', 'zIndex':2001}
        )
], className="container")


@app.callback(Output('my-interval', 'disabled'),
              Input('continue', 'n_clicks'),
              Input('stop', 'n_clicks'),
              Input('init', 'data'),
              Input('alpha', 'value'),
              Input('beta', 'value'),
              Input('w_line', 'value'),
              Input('w_edge', 'value'),
              Input('img-store', 'data'),
             Input('sigma', 'value'),
              Input('points', 'value'),
              prevent_initial_call=True
              )
def update_slider(con, stop, n, alpha, beta, w_line, w_edge, img, p,s):
    ctx = dash.callback_context
    if 'stop' in str(ctx.triggered_prop_ids) or \
            'alpha' in str(ctx.triggered_prop_ids) or \
            'beta' in str(ctx.triggered_prop_ids) or \
            'w_line' in str(ctx.triggered_prop_ids) or \
            'img-store' in str(ctx.triggered_prop_ids) or \
            'sigma' in str(ctx.triggered_prop_ids) or \
            'points' in str(ctx.triggered_prop_ids) or \
            'w_edge' in str(ctx.triggered_prop_ids):
        return True
    if 'continue' in str(ctx.triggered_id) or 'init' in str(ctx.triggered_id) or 'stop' in str(ctx.triggered_id):
        return False


@app.callback(Output('s-conture', 'value'),
              Output('s-conture', 'max'),
              Output('graphs', 'data'),
              Output('hidden-slider', 'style'),
              Input('my-interval', 'n_intervals'),
              Input('s-conture', 'value'),
              State('graphs', 'data'),
              Input('init', 'data'),
              State('alpha', 'value'),
              State('beta', 'value'),
              State('w_line', 'value'),
              State('w_edge', 'value'),
              State('img-store', 'data'),
              State('boundary_condition', 'data'),
              State('sigma', 'value'),
              prevent_initial_call=True
              )
def update_slider(n, slider_value, data, init, alpha, beta, w_line, w_edge, img, boundary_condition, sigma):
    ctx = dash.callback_context
    if 'init' in str(ctx.triggered_id):
        data = [init]
        slider_value = 1
    if 'my-interval' in str(ctx.triggered_id) or 'init' in str(ctx.triggered_id):
        if int(slider_value + 1) > len(data):
            img = gaussian(np.array(img), sigma, preserve_range=False)
            data.append(active_contour(img,np.array(data[-1]), gamma=0.001, alpha=alpha, beta=beta,  w_edge=w_edge, w_line=w_line,
                                    boundary_condition=boundary_condition, max_num_iter=5))
            max_v = int(slider_value) + 1
        else:
            max_v = dash.no_update
        return int(slider_value) + 1, max_v, data, {'width': '20%','visibility': 'visible'}
    else:
        return int(slider_value), dash.no_update, dash.no_update, {'width': '20%', 'visibility': 'visible'}


def upload_container(name):
    return [html.Div(html.P(name),
                     style={'height': '75%', 'textAlign': 'center', 'width': '80%', 'margin': '10px'}),
            dcc.Upload(
                [],
                id='img-upload',
                style={'visibility': 'hidden'},
            ),
            dbc.Button(className="bi bi-x-octagon-fill me-2", id='close',
                       style={'height': '75%', 'width': '20%', 'border': '1px solid', 'borderRadius': '5px',
                              'margin': '10px'})]


def upload_empty():
    return [
        dcc.Upload(
            children=html.P(['Upload Image']),
            id='img-upload',
            style={'width': '100%', 'border': '1px solid', 'borderRadius': '5px', 'textAlign': 'center'},
        ),
        dbc.Button(id='close', style={'visibility': 'hidden'})
    ]


@app.callback(Output('picture', 'figure'),
              State('graphs', 'data'),
              Input('s-conture', 'value'),
              Input('img-store', 'data'),
              State('picture', 'figure'),
              Input('sigma', 'value'),
              Input('picture', 'relayoutData'),
              prevent_initial_call=True
              )
def update_graph(graph, current_count, img, old_graph, sigma, n):
    ctx = dash.callback_context
    if 'relayoutData' in str(ctx.triggered_prop_ids):
        if 'shapes' in n.keys():
            if len(n['shapes']) > 1:
                old_graph['layout']['shapes'] = n['shapes'][1:]
            return old_graph
        else:
            return dash.no_update

    if 'img-store' in str(ctx.triggered_id):
        if img:
            img = gaussian(np.array(img), sigma, preserve_range=False)
            fig = px.imshow(img, color_continuous_scale='gray')
            fig.update_layout(newshape_line_color='cyan')
            fig['layout']['yaxis']['range'] = [float(img.shape[0]), float(0)]
            fig['layout']['yaxis'].update(autorange = True)
            fig['layout']['xaxis']['range'] = [float(0), float(img.shape[1])]
            fig['layout']['xaxis'].update(autorange = True)
            return fig
        else:
            return dash.no_update
    if 'sigma' in str(ctx.triggered_id):
        img = gaussian(np.array(img), sigma, preserve_range=False)
        img = px.imshow(img, color_continuous_scale='gray').to_plotly_json()
        old_graph['data'][0] = img['data'][0]
        return old_graph

    if len(old_graph['data']) > 1:
        old_graph['data'].pop()

    if 's-conture' in str(ctx.triggered_id):
        current_count -= 1
        a = go.Scatter(x=np.array(graph[current_count])[:, 1], y=np.array(graph[current_count])[:, 0],
                       line=go.scatter.Line(color='red'))
    # let it run if graphs object has changed

        old_graph['data'].append(a.to_plotly_json())
    old_graph['layout']['yaxis']['range'] = [float(np.array(img).shape[0]), float(0)]
    old_graph['layout']['yaxis']['autorange'] = False
    old_graph['layout']['xaxis']['range'] = [float(0), float(np.array(img).shape[1])]
    old_graph['layout']['xaxis']['autorange'] = False
    return old_graph





@app.callback(
    Output('img-store', 'data'),
    Output('upload_container', 'children'),
    Output('hidden-div', 'style'),
    Input('img-upload', 'contents'),
    State('img-upload', 'filename'),
    Input('close', 'n_clicks'),
    prevent_initial_call=True
)
def upload(contents, fname, n):
    ctx = dash.callback_context
    if 'close' in str(ctx.triggered_id):
        return [], upload_empty(), {'width': '100%', 'opacity': '0'}
    content_type, content_string = contents.split(',')
    decoded = base64.b64decode(content_string)
    img = skimage.io.imread(io.BytesIO(decoded))

    if len(img.shape) > 2:
        img = rgb2gray(img)
    else:
        img = img/np.max(img)
    return img, upload_container(fname), {'width': '100%', 'opacity': '1'}


def resample_polygon(yx, n_points):
    ### TODO
    # Hier könnte dein Code stehen b)
    ###
    return None



if __name__ == '__main__':
    app.run_server(debug=True, use_reloader=True)
