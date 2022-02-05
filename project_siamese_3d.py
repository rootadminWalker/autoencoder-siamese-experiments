import argparse
import pickle
from time import sleep

import plotly.graph_objs as go
from dash import dash
from dash import dcc
from dash import html
from dash.dependencies import Output, Input

arguments = {}
plots = {}
app = dash.Dash(__name__)
epoch = 1
reverse = False


@app.callback(
    Output('live-graph', 'figure'),
    [Input('graph-update', 'n_intervals')]
)
def update(n):
    global arguments, epoch, reverse
    color_map = {
        0: 'red',
        1: 'blue',
        2: 'gray',
        3: 'purple',
        4: 'orange',
        5: 'cyan',
        6: 'green',
        7: 'brown',
        8: 'pink',
        9: 'olive'
    }
    data = plots[epoch]
    print(f"Epoch: {epoch}")

    points = plots['points']

    xmin = -1.8
    ymin = -2
    zmin = -2
    xmax = 2.5
    ymax = 2
    zmax = 2

    scatters3ds = []
    for label, idxes in data['label_idxes'].items():
        color = color_map[label]
        point = points[idxes]
        scatters3ds.append(go.Scatter3d(
            x=point[:, 0],
            y=point[:, 1],
            z=point[:, 2],
            marker=go.scatter3d.Marker(size=3, color=color),
            mode='markers'
        ))

    if epoch >= len(plots.keys()):
        reverse = True
    elif epoch <= 1:
        reverse = False

    epoch += ((1 - reverse) * 1 + reverse * -1)

    return {
        'data': scatters3ds,
        'layout': go.Layout(
            title=f'Siamese clusters (epoch {epoch},embedding_dim=4)',
            # scene=go.layout.Scene(
            #     xaxis=go.layout.scene.XAxis(range=[xmin, xmax]),
            #     yaxis=go.layout.scene.YAxis(range=[ymin, ymax]),
            #     zaxis=go.layout.scene.ZAxis(range=[zmin, zmax])
            # ),
            height=800
        )
    }


def main(args):
    global plots, arguments
    arguments.update(args)
    with open(arguments['input_file'], 'rb') as p:
        plots = pickle.load(p)

    app.layout = html.Div(
        [
            dcc.Graph(id='live-graph', animate=True, animation_options={"frame": {"redraw": True}}),
            dcc.Interval(
                id='graph-update',
                interval=arguments['delay'],
                n_intervals=0
            ),
        ]
    )
    sleep(1)
    app.run_server(debug=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input-file', type=str, required=True,
                        help='Path to your scatters pickle')
    parser.add_argument('-d', '--delay', type=float, default=2,
                        help='Delay for screen refreshing')
    args = vars(parser.parse_args())
    main(args)
