import sys

from dash import Dash, html, dcc, callback, Output, Input
import plotly.express as px
import plotly.graph_objects as go
import pandas as pd
from sklearn.metrics import roc_auc_score, roc_curve, auc

from evaluation.presentation.load import load
from evaluation.threshold import get_threshold

# Incorporate data
df = load()

app = Dash()

# Requires Dash 2.17.0 or later
app.layout = [
    html.Div(className='row', children="Test", style={'padding': 10, 'fontSize': 30, 'textAlign': 'center'}),
    html.Div(className='row', children=[
        dcc.RadioItems(options=['GPT-4o-mini', 'Llama-3.3-70B-Instruct'],
                       value='Llama-3.3-70B-Instruct', id='controls-generative_model'),
        dcc.RadioItems(options=['detect-gpt', 'fast-detect-gpt'],
                       value='detect-gpt', id='controls-detector'),
    ], style={'padding': 10, 'fontSize': 10, 'textAlign': 'center'}),
    html.Div(className='row', children=[
        dcc.Slider(0, 4,
                   step=None,
                   marks={
                       0: 'Human',
                       1: 'Improve-Human',
                       2: 'Rewrite-Human',
                       3: 'Summary',
                       4: 'Task+Summary'
                   },
                   value=0,
                   id='controls-boundary'
                   ),
    ], style={'padding': 10, 'fontSize': 10, 'textAlign': 'center', 'width': '60%', 'margin': '0 auto'}),
    dcc.Graph(figure={}, id='controls-and-graph'),
    dcc.Graph(figure={}, id='roc-auc')
]


# Add controls to build the interaction
@callback(
    Output(component_id='controls-and-graph', component_property='figure'),
    Output(component_id='roc-auc', component_property='figure'),
    Input(component_id='controls-generative_model', component_property='value'),
    Input(component_id='controls-detector', component_property='value'),
    Input(component_id='controls-boundary', component_property='value'),
)
def update_graph(generative_model, detector, boundary):
    sub_df = df[(df.generative_model == generative_model) | (df.generative_model == "Human")]
    sub_df = sub_df[sub_df.detector == detector]

    # fig = px.histogram(sub_df, x='detector', y="prediction", histfunc='avg')
    desired_order = ["Human", "Improved-Human", "Rewrite-Human", "Summary", "Task+Summary", "Task", "Rewrite-LLM",
                     "Dipper"]

    for i in range(boundary):
        sub_df['is_human'][sub_df.prompt_mode == desired_order[i + 1]] = True

    fig = px.violin(sub_df, x="prompt_mode", y="prediction", color="is_human")
    fig.update_xaxes(categoryorder='array', categoryarray=desired_order)
    fig.update_layout(violingap=0, violinmode='overlay')
    fig.update_traces(meanline_visible=True)

    results = []
    for mode in sub_df.prompt_mode.unique():
        if mode == "Human":
            continue

        human_label = sub_df[sub_df.prompt_mode == mode]["is_human"].unique()
        if len(human_label) != 1:
            sys.exit("Not unique label")
        else:
            human_label = human_label[0]

        sub_sub_df = sub_df[(sub_df.prompt_mode == mode) | (sub_df.prompt_mode == ("Task" if human_label else "Human"))]

        fpr, tpr, _ = roc_curve(~sub_sub_df['is_human'], sub_sub_df['prediction'], drop_intermediate=False)

        results.append({
            'mode': mode,
            'fpr': fpr,
            'tpr': tpr,
        })
    threshold = get_threshold(sub_df)
    fig.add_shape(
        type='line',
        x0=0, x1=7,
        y0=threshold, y1=threshold,
        line=dict(color='Gray', dash='dash'),
    )

    sub_df = pd.DataFrame(results)
    sub_df = sub_df.explode(['fpr', 'tpr'], ignore_index=True)

    fig2 = px.line(sub_df, x="fpr", y="tpr", color="mode")
    fig2.update_xaxes(categoryorder='array', categoryarray=desired_order)
    return fig, fig2


if __name__ == '__main__':
    app.run(debug=True)
